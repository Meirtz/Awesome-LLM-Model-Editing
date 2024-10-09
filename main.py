import requests
from bs4 import BeautifulSoup
import os
import datetime
import time
import json
from openai import OpenAI
import subprocess

class LLMClassifier:
    def __init__(self, api_key=None, model="deepseek-chat", max_tokens=1024, temperature=0):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if self.api_key is None:
            raise ValueError("API key must be provided or set as an environment variable")
        
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        
        self.configure(model, max_tokens, temperature)
    
    def configure(self, model, max_tokens, temperature):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def classify(self, title, abstract, prompt_template, max_retries=10, retry_delay=5):
        prompt = prompt_template.format(title=title, abstract=abstract)
        retries = 0
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Always respond in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                answer = response.choices[0].message.content.strip()
                print(f"Raw answer: {answer}")
                
                try:
                    # 删除可能的 Markdown 代码块标记
                    answer = answer.replace('```json', '').replace('```', '').strip()
                    
                    # 尝试修复可能的 JSON 截断
                    if answer.endswith('...'):
                        answer = answer.rstrip('.')
                    if not answer.endswith('}'):
                        answer += '}'
                    
                    parsed_answer = json.loads(answer)
                    print(f"Parsed answer: {json.dumps(parsed_answer, indent=2)}")
                    
                    if 'is_relevant' in parsed_answer and 'explanation' in parsed_answer:
                        return parsed_answer['is_relevant'], parsed_answer['explanation']
                    else:
                        print(f"Unexpected JSON structure. Retrying... ({retries+1}/{max_retries})")
                        retries += 1
                        time.sleep(retry_delay)
                        continue
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON: {e}. Retrying... ({retries+1}/{max_retries})")
                    print(f"Problematic JSON: {answer}")
                    retries += 1
                    time.sleep(retry_delay)
                    continue
                
            except Exception as e:
                print(f"Service error: {e}, retrying in {retry_delay} seconds... ({retries+1}/{max_retries})")
                retries += 1
                time.sleep(retry_delay)
        
        raise Exception(f"Max retries reached. Could not classify title: {title}")

class PageParser:
    def extract_papers(self, url, max_retries=5, retry_delay=2):
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

class RepoManager:
    def __init__(self, config_path='config.json', prompt_path='prompts.json', template_path='readme_template.md'):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found. Please copy config.example.json to {config_path} and modify it.")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.repos = json.load(f)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompts = json.load(f)
        with open(template_path, 'r', encoding='utf-8') as f:
            self.template = f.read()
    
    def ensure_repo_exists(self, repo_name):
        repo_path = self.repos[repo_name]['path']
        if not os.path.exists(repo_path):
            print(f"Repository {repo_name} does not exist locally. Creating...")
            os.makedirs(repo_path)
            
            # 创建 .gitignore 文件
            gitignore_path = os.path.join(repo_path, '.gitignore')
            with open(gitignore_path, 'w') as f:
                f.write("# Python\n__pycache__/\n*.py[cod]\n\n# Environments\n.env\n.venv\nenv/\nvenv/\n\n# IDEs\n.vscode/\n.idea/\n")
            
            print(f"Created {repo_name} directory and added .gitignore")
            
            # 初始化 Git 仓库
            try:
                subprocess.run(['git', 'init'], cwd=repo_path, check=True)
                print(f"Initialized Git repository for {repo_name}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to initialize Git repository: {e}")
        return True

    def update_readme(self, repo_name, content):
        if not self.ensure_repo_exists(repo_name):
            print(f"Cannot update README for {repo_name}")
            return

        today = datetime.date.today()
        readme_content = self.template.format(
            repo_name=repo_name,
            topic=self.repos[repo_name]['topic'],
            date=today,
            content=content
        )
        readme_path = os.path.join(self.repos[repo_name]['path'], 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"Updated README for {repo_name}")

        self.commit_and_push(repo_name)

    def commit_and_push(self, repo_name):
        repo_path = self.repos[repo_name]['path']
        today = datetime.date.today()
        commit_message = f"Update README for {today}"

        try:
            # Git add
            subprocess.run(['git', 'add', 'README.md'], cwd=repo_path, check=True)
            print(f"Git add successful for {repo_name}")

            # Git commit
            subprocess.run(['git', 'commit', '-m', commit_message], cwd=repo_path, check=True)
            print(f"Git commit successful for {repo_name}")

            # Git push
            subprocess.run(['git', 'push'], cwd=repo_path, check=True)
            print(f"Git push successful for {repo_name}")

        except subprocess.CalledProcessError as e:
            print(f"Git operation failed for {repo_name}: {e}")

if __name__ == "__main__":
    classifier = LLMClassifier(model="deepseek-chat")
    parser = PageParser()
    repo_manager = RepoManager()

    repo_name = "LLM-Paper-Daily"
    repo_info = repo_manager.repos[repo_name]
    url = repo_info['url']
    papers = parser.extract_papers(url)
    num_of_papers = len(papers)
    print(f"Total papers in {repo_name}: {num_of_papers}")

    llm_related_papers = []

    prompt_template = repo_manager.prompts.get(repo_name)
    print(f"Using prompt template for {repo_name}: {prompt_template}")

    for i, paper in enumerate(papers):
        print(f"Paper {i+1}: {paper['title']}")
        title = paper['title']
        abstract = paper['abstract']
        print(f"({i+1}/{num_of_papers}) Examining paper: {title}")
        is_llm_related, explanation = classifier.classify(title, abstract, prompt_template=prompt_template)
        if is_llm_related:
            llm_related_papers.append((paper, explanation))
            if len(llm_related_papers) == 3:
                break  # Stop after finding 3 relevant papers

    content = ""
    for i, (paper, explanation) in enumerate(llm_related_papers, 1):
        content += f"### {i}. [{paper['title']}]({paper['link']})\n- **Authors**: {paper['authors']}\n- **Link**: {paper['link']}\n- **Explanation**: {explanation}\n\n"

    repo_manager.update_readme(repo_name, content)

    print(f"Updated README for {repo_name} with {len(llm_related_papers)} papers.")