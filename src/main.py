from llm_classifier import LLMClassifier
from page_parser import PageParser
from .repo_manager import RepoManager  # 修改这一行
from utils import extract_arxiv_id
from datetime import datetime, timedelta
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
        if prompt_template is None:
            print(f"Error: No prompt template found for {repo_name}. Skipping this repository.")
            continue
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
                    print(f"Exception details: {exc}")
                    print(f"Prompt template: {prompt_template}")

        # 从数据库获取所有论文
        all_papers = repo_manager.dbs[repo_name].get_papers()
        
        # 筛选出相关的论文
        relevant_papers = [p for p in all_papers if p.get('is_relevant', True)]
        
        # 按日期排序，最新的在前
        relevant_papers.sort(key=lambda x: datetime.strptime(x['date_added'], '%Y-%m-%d').date(), reverse=True)
        
        # 选择最近30天内的论文，最多100篇
        today = datetime.now().date()
        recent_papers = [
            p for p in relevant_papers 
            if (today - datetime.strptime(p['date_added'], '%Y-%m-%d').date()).days <= 30
        ][:100]

        # 打印最近论文的摘要统计信息
        print(f"Total relevant papers: {len(relevant_papers)}")
        print(f"Recent papers (last 30 days, max 100): {len(recent_papers)}")
        if recent_papers:
            print(f"Date range of recent papers: from {recent_papers[-1]['date_added']} to {recent_papers[0]['date_added']}")
        else:
            print("No recent papers found.")

        # 使用最近的论文更新README
        repo_manager.update_readme(repo_name, recent_papers)

        print(f"Updated README for {repo_name} with {len(recent_papers)} papers.")
        print(f"Processed {len(relevant_papers)} new relevant papers.")
        print(f"Processed {len(irrelevant_papers)} irrelevant papers.")
        print(f"Skipped {skipped_papers} already processed papers.")

if __name__ == "__main__":
    main(["LLM-Paper-Daily"]) # only for test