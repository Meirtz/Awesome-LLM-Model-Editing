import re

def extract_arxiv_id(link: str) -> str:
    match = re.search(r'(\d+\.\d+)', link)
    return match.group(1) if match else ''