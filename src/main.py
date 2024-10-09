from llm_classifier import LLMClassifier
from page_parser import PageParser
from repo_manager import RepoManager
from utils import extract_arxiv_id
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
from tqdm import tqdm

# 创建一个线程本地存储对象来存储每个线程的LLMClassifier实例
thread_local = threading.local()

def get_classifier():
    if not hasattr(thread_local, "classifier"):
        thread_local.classifier = LLMClassifier(model="deepseek-chat")
    return thread_local.classifier

def process_paper(paper, repo_info, prompt_template, db):
    classifier = get_classifier()
    title = paper['title']  # 标题已经在爬虫阶段清理过了
    abstract = paper['abstract']
    arxiv_id = extract_arxiv_id(paper['link'])
    
    if not db.paper_exists(title, arxiv_id):
        is_relevant, explanation = classifier.classify(title, abstract, prompt_template=prompt_template)
        if is_relevant:
            paper['keywords'] = explanation
            paper['category'] = repo_info['topic']
            paper['arxiv_id'] = arxiv_id
            paper['summary'] = classifier.generate_summary(title, abstract)
            paper['date_added'] = datetime.now().date()
            db.add_paper(paper)
            return paper, True
        else:
            db.add_irrelevant_paper(title, arxiv_id, paper['link'])
            return {'title': title, 'link': paper['link']}, False
    else:
        return None, None

def main(repos_to_update):
    parser = PageParser()
    repo_manager = RepoManager()

    for repo_name in repos_to_update:
        if repo_name not in repo_manager.repos:
            print(f"Repository {repo_name} not found in config. Skipping.")
            continue

        repo_info = repo_manager.repos[repo_name]
        if not repo_info.get('enabled', True):
            print(f"Skipping {repo_name} as it is disabled in the config.")
            continue

        repo_path = os.path.join('./repositories', repo_name)
        url = repo_info['url']
        papers = parser.extract_papers(url)
        num_of_papers = len(papers)
        print(f"Total papers in {repo_name}: {num_of_papers}")

        prompt_template = repo_manager.prompts.get(repo_name)
        print(f"Using prompt template for {repo_name}: {prompt_template}")

        relevant_papers = []
        irrelevant_papers = []
        skipped_papers = 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(process_paper, paper, repo_info, prompt_template, repo_manager.dbs[repo_name]) for paper in papers]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing papers"):
                try:
                    result, is_relevant = future.result()
                    if result:
                        if is_relevant:
                            relevant_papers.append(result)
                            print(f"Paper processed and added as relevant: {result['title']}")
                        else:
                            irrelevant_papers.append(result)
                            print(f"Paper processed and added as irrelevant: {result['title']}")
                    else:
                        skipped_papers += 1
                        print(f"Paper already exists and skipped: {papers[futures.index(future)]['title']}")
                except Exception as exc:
                    print(f"Paper processing generated an exception: {papers[futures.index(future)]['title']}")
                    print(f"{exc}")

        # 使用当前处理的论文更新README
        repo_manager.update_readme(repo_name, relevant_papers)

        print(f"Updated README for {repo_name} with {len(relevant_papers)} papers.")
        print(f"Processed {len(irrelevant_papers)} irrelevant papers.")
        print(f"Skipped {skipped_papers} already processed papers.")

if __name__ == "__main__":
    main(["LLM-Paper-Daily"])