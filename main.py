from llm_classifier import LLMClassifier
from page_parser import PageParser
from repo_manager import RepoManager
from utils import extract_arxiv_id

def main():
    classifier = LLMClassifier(model="deepseek-chat")
    parser = PageParser()
    repo_manager = RepoManager()

    for repo_name, repo_info in repo_manager.repos.items():
        if not repo_info.get('enabled', True):
            print(f"Skipping {repo_name} as it is disabled in the config.")
            continue

        url = repo_info['url']
        papers = parser.extract_papers(url)
        num_of_papers = len(papers)
        print(f"Total papers in {repo_name}: {num_of_papers}")

        relevant_papers = []
        prompt_template = repo_manager.prompts.get(repo_name)
        print(f"Using prompt template for {repo_name}: {prompt_template}")

        for i, paper in enumerate(papers):
            print(f"Paper {i+1}: {paper['title']}")
            title = paper['title']
            abstract = paper['abstract']
            arxiv_id = extract_arxiv_id(paper['link'])
            print(f"({i+1}/{num_of_papers}) Examining paper: {title}")
            
            if not repo_manager.dbs[repo_name].paper_exists(title, arxiv_id):
                is_relevant, explanation = classifier.classify(title, abstract, prompt_template=prompt_template)
                if is_relevant:
                    paper['keywords'] = explanation
                    paper['category'] = repo_info['topic']
                    paper['arxiv_id'] = arxiv_id
                    paper['summary'] = classifier.generate_summary(title, abstract)
                    repo_manager.dbs[repo_name].add_paper(paper)
                    relevant_papers.append(paper)
                    if len(relevant_papers) == 3:
                        break  # Stop after finding 3 relevant papers
            else:
                print(f"Paper already exists in database: {title}")

        repo_manager.update_readme(repo_name, relevant_papers)
        print(f"Updated README for {repo_name} with {len(relevant_papers)} papers.")

if __name__ == "__main__":
    main()