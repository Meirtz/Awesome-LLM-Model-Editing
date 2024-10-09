import json
import os
import subprocess
from typing import List, Dict, Any
from paper_database import PaperDatabase
from datetime import datetime, timedelta, date

class RepoManager:
    def __init__(self, config_path='config.json', prompt_path='prompts.json', template_path='readme_template.md'):
        # 获取当前脚本的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录（假设 src 目录在项目根目录下）
        root_dir = os.path.dirname(current_dir)
        
        # 构建完整的文件路径
        self.config_path = os.path.join(root_dir, config_path)
        self.prompt_path = os.path.join(root_dir, prompt_path)
        self.template_path = os.path.join(root_dir, template_path)
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found. Please copy config.example.json to {self.config_path} and modify it.")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.repos = json.load(f)
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            self.prompts = json.load(f)
        with open(self.template_path, 'r', encoding='utf-8') as f:
            self.template = f.read()
        
        self.dbs = {}
        self.repos_dir = './repositories'
        for repo_name, repo_info in self.repos.items():
            db_path = os.path.join(repo_info['path'], f'{repo_name.lower().replace(" ", "_")}.db')
            try:
                self.dbs[repo_name] = PaperDatabase(db_path)
            except Exception as e:
                print(f"Error initializing database for {repo_name}: {e}")
                raise
    
    def ensure_repo_exists(self, repo_name):
        repo_path = os.path.join(self.repos_dir, repo_name)
        if not os.path.exists(repo_path):
            print(f"Repository {repo_name} does not exist locally. Creating...")
            os.makedirs(repo_path, exist_ok=True)
            
            gitignore_path = os.path.join(repo_path, '.gitignore')
            with open(gitignore_path, 'w') as f:
                f.write("# Python\n__pycache__/\n*.py[cod]\n\n# Environments\n.env\n.venv\nenv/\nvenv/\n\n# IDEs\n.vscode/\n.idea/\n\n# Database\n*.db\n")
            
            print(f"Created {repo_name} directory and added .gitignore")
            
            try:
                subprocess.run(['git', 'init'], cwd=repo_path, check=True)
                print(f"Initialized Git repository for {repo_name}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to initialize Git repository: {e}")
        return True

    def update_readme(self, repo_name: str, recent_papers: List[Dict[str, Any]]):
        if not self.ensure_repo_exists(repo_name):
            print(f"Cannot update README for {repo_name}")
            return

        today = datetime.now().date()

        content = ""
        for i, paper in enumerate(recent_papers, 1):
            content += f"### {i}. [{paper['title']}]({paper['link']})\n\n"
            content += f"**Summary**: {paper['summary']}\n\n"

        readme_content = self.template.format(
            repo_name=repo_name,
            topic=self.repos[repo_name]['topic'],
            date=today,
            content=content
        )
        readme_path = os.path.join(self.repos_dir, repo_name, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"Updated README for {repo_name} with {len(recent_papers)} papers.")

        self.commit_and_push(repo_name)

    def commit_and_push(self, repo_name):
        repo_path = self.repos[repo_name]['path']
        today = date.today()
        commit_message = f"Update README for {today}"

        try:
            subprocess.run(['git', 'add', 'README.md'], cwd=repo_path, check=True)
            print(f"Git add successful for {repo_name}")

            subprocess.run(['git', 'commit', '-m', commit_message], cwd=repo_path, check=True)
            print(f"Git commit successful for {repo_name}")

            subprocess.run(['git', 'push'], cwd=repo_path, check=True)
            print(f"Git push successful for {repo_name}")

        except subprocess.CalledProcessError as e:
            print(f"Git operation failed for {repo_name}: {e}")

    def get_prompt(self, repo_name):
        return self.prompts.get(repo_name, self.prompts.get('default'))