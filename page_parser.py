import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict

class PageParser:
    def extract_papers(self, url, max_retries=5, retry_delay=2) -> List[Dict]:
        papers = []
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                paper_entries = soup.find_all('dt')

                for entry in paper_entries:
                    meta_div = entry.find_next('dd').find('div', class_='meta')
                    title_element = meta_div.find('div', class_='list-title')
                    title = title_element.text.replace('Title: ', '').strip()
                    authors_element = meta_div.find('div', class_='list-authors')
                    authors = ', '.join(author.text for author in authors_element.find_all('a'))
                    abstract_element = meta_div.find('p', class_='mathjax')
                    try:
                        abstract = abstract_element.text.strip()
                    except Exception as e:
                        print(f"Error: {e}. Abstract not found, ending extraction.")
                        break
                    pdf_link_element = entry.find('a', title='Download PDF')
                    pdf_link = 'https://arxiv.org' + pdf_link_element['href'] if pdf_link_element else None
                    papers.append({'title': title,
                                   'authors': authors,
                                   'abstract': abstract,
                                   'link': pdf_link})
                return papers

            except (requests.exceptions.RequestException, Exception) as e:
                print(f"Error: {e}. Retrying {retries + 1}/{max_retries}...")
                retries += 1
                time.sleep(retry_delay)
        print(f"Failed to extract papers after {max_retries} retries.")
        return []