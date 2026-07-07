from datetime import datetime
import arxiv
from scrape_credibility import get_score
from concurrent.futures import ThreadPoolExecutor, as_completed

ARVIX_FOUNDING_TIME = 19910101000000
MAX_WORKERS=5

def get_latest_microstructure_papers(n: int = 100, before_date:str | None = None) -> list[dict]:
    """
    Fetch the latest `n` papers from arXiv q-fin.TR
    (Trading and Market Microstructure), returning title + authors.
    """
    if before_date == None:
        before_date = datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        before_date = datetime.strptime(before_date, '%Y-%m-%d').strftime("%Y%m%d%H%M%S")
    client = arxiv.Client(page_size=n, delay_seconds=1)
    search = arxiv.Search(
        query=f"cat:q-fin.TR AND submittedDate:[{ARVIX_FOUNDING_TIME} TO {before_date}]",
        max_results=n,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = []
    for paper in client.results(search):
        results.append({
            "title": paper.title,
            "published": paper.published.date().isoformat(),
            "authors": [str(a) for a in paper.authors],
            "arxiv_id": paper.get_short_id(),
            "url": paper.entry_id,
        })

    return results

def score_papers(papers: list[dict]) -> tuple[list[dict], int, int]:
    success = 0
    total = 0
    paper_score = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as excecutor:
        title_async = {
            excecutor.submit(get_score, paper['title']): paper
            for paper in papers
        }

        for i, future in enumerate(as_completed(title_async), 1):
            paper = title_async[future]
            result = future.result()
            if isinstance(result, int):
                paper['Score'] = result
                paper_score.append(paper)
                success += 1
            total += 1
    return paper_score, success, total

def find_and_score_papers(n=100, before_date:str | None = None) -> tuple[list[dict], int, int]:
    papers = get_latest_microstructure_papers(n, before_date=before_date)
    scored_papers, success, total = score_papers(papers)
    return scored_papers, success, total