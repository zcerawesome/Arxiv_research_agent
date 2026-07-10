"""
arXiv Paper Credibility Report
================================
Given an arXiv paper title (and optionally a specific author name),
this script queries OpenAlex to produce a full credibility report
covering: author citation metrics, institution, peer-review status,
and an overall credibility score.

Usage:
    python arxiv_credibility.py

Requirements:
    pip install requests
"""

import requests
from difflib import SequenceMatcher
from datetime import datetime
import time
import threading
import json
import sys

API_SLEEP = 1.0

_rate_lock = threading.Lock()
_last_call_time = 0.0

def _throttle():
    """Block until at least API_SLEEP seconds have passed since the last
    OpenAlex request, across all threads."""
    global _last_call_time
    with _rate_lock:
        wait = API_SLEEP - (time.monotonic() - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.monotonic()


def find_paper(title: str):
    """Search OpenAlex for a paper by title. Returns the best match or None."""
    _throttle()
    # OpenAlex's filter mini-language treats unescaped commas as filter
    # separators (raising a 400), so quote the value to search it literally.
    safe_title = title.replace('"', "'")
    response = requests.get(
        "https://api.openalex.org/works",
        params={
            "filter": f'title.search:"{safe_title}"',
            "per_page": 5,
            "select": "id,title,authorships,cited_by_count,publication_year,"
                      "open_access,primary_location,is_retracted"
        }
    )
    if response.status_code != 200:
        print(f"[ERROR] OpenAlex works API returned {response.status_code}")
        return None
    results = response.json().get("results", [])
    if not results:
        return None

    best, best_score = None, 0.0
    for paper in results:
        score = SequenceMatcher(
            None,
            title.lower(),
            (paper.get("title") or "").lower()
        ).ratio()
        if score > best_score:
            best, best_score = paper, score

    return best if best_score >= 0.75 else None



def fetch_author_metrics(openalex_author_url: str) -> dict:
    """Fetch full author record from OpenAlex including citation stats."""
    _throttle()
    # author_stub["id"] is the openalex.org landing-page URL, not the API
    # endpoint — hitting it directly returns a 403 HTML page, not JSON.
    api_url = openalex_author_url.replace("https://openalex.org/", "https://api.openalex.org/authors/")
    response = requests.get(api_url)
    if response.status_code != 200:
        return {}
    return response.json()


def extract_authors(paper: dict, target_name: str | None = None) -> list[dict]:
    """
    Pull author + institution data from a paper's authorship list.
    If target_name is given, only return matching authors.
    """
    authors = []

    for authorship in paper.get("authorships", []):
        author_stub = authorship.get("author", {})
        name = author_stub.get("display_name", "Unknown")

        # Filter to target author if specified
        if target_name and target_name.lower() not in name.lower():
            continue

        # Fetch full metrics for this author
        author_url = author_stub.get("id")
        full_metrics = fetch_author_metrics(author_url) if author_url else {}
        summary = full_metrics.get("summary_stats", {})

        # Institution(s) listed on this specific paper
        institutions = [
            {
                "name": inst.get("display_name", "Unknown"),
                "country": inst.get("country_code", "N/A"),
                "type": inst.get("type", "N/A"),
                "ror": inst.get("ror", "N/A")
            }
            for inst in authorship.get("institutions", [])
        ]

        authors.append({
            "name": name,
            "position": authorship.get("author_position", "unknown"),
            "orcid": author_stub.get("orcid") or "Not available",
            "openalex_id": author_url,
            "institutions": institutions if institutions else [{"name": "Not listed"}],
            "works_count": full_metrics.get("works_count", 0),
            "cited_by_count": full_metrics.get("cited_by_count", 0),
            "h_index": summary.get("h_index", 0),
            "i10_index": summary.get("i10_index", 0),
            "2yr_mean_citedness": summary.get("2yr_mean_citedness", 0.0),
        })

    return authors



def score_author(author: dict) -> dict:
    """Compute a 0-100 credibility score for a single author."""
    score = 0
    reasons = []

    # h-index (max 35 pts)
    h = author.get("h_index", 0)
    if h >= 30:
        score += 35; reasons.append(f"h-index {h} (elite)")
    elif h >= 15:
        score += 25; reasons.append(f"h-index {h} (strong)")
    elif h >= 5:
        score += 15; reasons.append(f"h-index {h} (emerging)")
    else:
        score += 5;  reasons.append(f"h-index {h} (limited history)")

    cited = author.get("cited_by_count", 0)
    if cited >= 10000:
        score += 25; reasons.append(f"{cited:,} total citations (highly influential)")
    elif cited >= 1000:
        score += 18; reasons.append(f"{cited:,} total citations (well cited)")
    elif cited >= 100:
        score += 10; reasons.append(f"{cited:,} total citations (moderate)")
    else:
        score += 3;  reasons.append(f"{cited:,} total citations (low)")

    # Recent citation velocity — 2yr mean (max 20 pts)
    recent = author.get("2yr_mean_citedness", 0)
    if recent >= 500:
        score += 20; reasons.append(f"{recent:.1f} avg citations/paper (last 2yr, very active)")
    elif recent >= 50:
        score += 12; reasons.append(f"{recent:.1f} avg citations/paper (last 2yr, active)")
    elif recent >= 5:
        score += 6;  reasons.append(f"{recent:.1f} avg citations/paper (last 2yr, moderate)")
    else:
        reasons.append("Low recent citation activity")

    # Publication volume (max 10 pts)
    works = author.get("works_count", 0)
    if works >= 20:
        score += 10; reasons.append(f"{works} total publications (established)")
    elif works >= 5:
        score += 5;  reasons.append(f"{works} total publications (early career)")
    else:
        reasons.append(f"Only {works} publication(s) on record")

    # ORCID verified (max 10 pts)
    if author.get("orcid") and author["orcid"] != "Not available":
        score += 10; reasons.append("ORCID verified identity")

    return {"score": min(score, 100), "reasons": reasons}



def score_paper(paper: dict) -> dict:
    """Score paper-level credibility signals."""
    score = 0
    reasons = []

    # Citation count of the paper itself, normalized by age. A paper
    # published this year hasn't had time to accrue citations the way an
    # older paper has, so raw counts unfairly punish recent work.
    cited = paper.get("cited_by_count", 0)
    year = paper.get("publication_year")
    age_years = (datetime.now().year - year) if year else None

    if age_years is not None and age_years <= 0:
        score += 20
        reasons.append(f"Paper cited {cited:,} times (published this year — too recent to judge by citations)")
    else:
        rate = cited / age_years if age_years else cited
        if rate >= 50:
            score += 35; reasons.append(f"Paper cited {cited:,} times (~{rate:.1f}/yr — highly influential)")
        elif rate >= 10:
            score += 25; reasons.append(f"Paper cited {cited:,} times (~{rate:.1f}/yr — well received)")
        elif rate >= 2:
            score += 15; reasons.append(f"Paper cited {cited:,} times (~{rate:.1f}/yr — gaining traction)")
        else:
            score += 5;  reasons.append(f"Paper cited {cited:,} times (~{rate:.1f}/yr — limited reception so far)")

    # Published in a peer-reviewed venue
    location = paper.get("primary_location") or {}
    source = location.get("source") or {}
    venue = source.get("display_name", "")
    is_oa_journal = source.get("type") in ("journal", "conference")

    if venue and is_oa_journal:
        score += 35
        reasons.append(f"Published in peer-reviewed venue: {venue}")
    elif venue:
        score += 15
        reasons.append(f"Associated venue: {venue} (type: {source.get('type', 'unknown')})")
    else:
        reasons.append("No peer-reviewed venue found — likely preprint only")

    # Open access
    oa = paper.get("open_access") or {}
    if oa.get("is_oa"):
        score += 15
        reasons.append("Open access — freely verifiable")

    # Retraction flag
    if paper.get("is_retracted"):
        score = 0
        reasons = ["  PAPER HAS BEEN RETRACTED — do not use as credible source"]

    # Publication year context
    year = paper.get("publication_year")
    if year:
        reasons.append(f"Published: {year}")

    return {"score": min(score, 100), "reasons": reasons}



def credibility_report(paper_title: str, author_name: str | None = None) -> dict:
    """
    Full credibility report for an arXiv paper.
    Returns structured dict with paper info, per-author scores, and overall score.
    """

    # Find paper
    paper = find_paper(paper_title)
    if not paper:
        return {"error": f"Could not find paper matching: '{paper_title}'"}


    # Score the paper
    paper_scores = score_paper(paper)

    # Extract and score each author
    authors = extract_authors(paper, target_name=author_name)
    if not authors:
        return {
            "error": "No matching authors found on this paper.",
            "paper_title": paper.get("title")
        }

    author_reports = []
    for author in authors:
        author_score = score_author(author)
        author_reports.append({
            **author,
            "credibility_score": author_score["score"],
            "score_reasons": author_score["reasons"]
        })

    # Overall score: weighted average of paper (40%) + avg author score (60%)
    avg_author_score = sum(a["credibility_score"] for a in author_reports) / len(author_reports)
    overall = round(0.4 * paper_scores["score"] + 0.6 * avg_author_score)

    return {
        "paper_title": paper.get("title"),
        "publication_year": paper.get("publication_year"),
        "paper_citations": paper.get("cited_by_count", 0),
        "is_retracted": paper.get("is_retracted", False),
        "paper_score": paper_scores["score"],
        "paper_score_reasons": paper_scores["reasons"],
        "authors": author_reports,
        "overall_credibility_score": overall,
        "overall_grade": grade(overall)
    }


def grade(score: int) -> str:
    if score >= 80: return "A"
    if score >= 65: return "B"
    if score >= 45: return "C"
    if score >= 25: return "D"
    return "F"



def print_report(report: dict):
    if "error" in report:
        print(f"\n Error: {report['error']}")
        return

    print("=" * 60)
    print(f"  CREDIBILITY REPORT")
    print("=" * 60)
    print(f"  Paper : {report['paper_title']}")
    print(f"  Year  : {report.get('publication_year', 'N/A')}")
    print(f"  Cited : {report['paper_citations']:,} times")
    if report.get("is_retracted"):
        print("    RETRACTED")
    print()

    print(f"   Paper Score: {report['paper_score']}/100")
    for reason in report["paper_score_reasons"]:
        print(f"     • {reason}")
    print()

    for author in report["authors"]:
        print(f"   {author['name']} ({author['position']} author)")
        for inst in author["institutions"]:
            name = inst.get("name", "Unknown")
            country = inst.get("country", "")
            itype = inst.get("type", "")
            print(f"     Institution : {name} ({country}) — {itype}")
        print(f"     ORCID       : {author['orcid']}")
        print(f"     h-index     : {author['h_index']}")
        print(f"     Total cited : {author['cited_by_count']:,}")
        print(f"     Works       : {author['works_count']}")
        print(f"     2yr citedness: {author['2yr_mean_citedness']:.1f}")
        print(f"     Score       : {author['credibility_score']}/100")
        for reason in author["score_reasons"]:
            print(f"       ✓ {reason}")
        print()

    print("─" * 60)
    print(f"  OVERALL SCORE : {report['overall_credibility_score']}/100")
    print(f"  GRADE         : {report['overall_grade']}")
    print("=" * 60)

def get_score(paper_title, author=None):
    report = credibility_report(paper_title, author)
    return report.get('overall_credibility_score', 'ERROR Failed to load paper')


if __name__ == "__main__":
    PAPER_TITLE  = "Attention Is All You Need"
    AUTHOR_NAME  = None 

    print(get_score(PAPER_TITLE, AUTHOR_NAME))