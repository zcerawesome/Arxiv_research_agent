import pymupdf
import requests
import io
import re


def extract_section(pdf_stream: io.BytesIO, start: str | list[str], end: str | list[str]) -> str | None:
    """
    Extract a section from a PDF using pymupdf text extraction, from a
    heading matching any of `start` up to a heading matching any of `end`.
    `start`/`end` accept a single keyword or a list of alternatives
    (e.g. start=["conclusion", "concluding remarks", "summary"]).
    Returns None if no matching start heading is found.
    """
    doc = pymupdf.open(stream=pdf_stream, filetype='pdf')
    text = "\n".join(page.get_text() for page in doc)
    doc.close()

    def heading(keywords: str | list[str]) -> str:
        if isinstance(keywords, str):
            keywords = [keywords]
        alternation = "|".join(re.escape(k) for k in keywords)

        # return the keywords used for the start and end to extract subsections
        # from research pdf

        return rf"^\s*(?:[\w.-]+\s+){{0,6}}(?:{alternation})\w*\s*$"

    pattern = re.compile(
        rf"(?im){heading(start)}\n(.*?)(?={heading(end)}|\Z)",
        re.DOTALL,
    )

    match = pattern.search(text)
    return match.group(1).strip() if match else None

def pdf_to_text(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def stream_to_text(pdf_stream: io.BytesIO) -> str:
    doc = pymupdf.open(stream=pdf_stream, filetype='pdf')
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def get_pdf_bytes(arxiv_id: str) -> io.BytesIO:
    # arxiv_id like "2301.12345" or "2301.12345v2"
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(url)
    response.raise_for_status()
    return io.BytesIO(response.content)