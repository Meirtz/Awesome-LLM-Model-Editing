import sqlite3
import os
import json

def clean_titles(db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # 获取所有论文
        cursor.execute('SELECT id, title FROM papers')
        papers = cursor.fetchall()
        
        # 清理标题并更新数据库
        for paper_id, title in papers:
            cleaned_title = title.replace("Title:", "").strip()
            cursor.execute('UPDATE papers SET title = ? WHERE id = ?', (cleaned_title, paper_id))
        
        conn.commit()
        print(f"Cleaned {len(papers)} titles in the database.")

def main():
    # 读取 config.json 文件
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    for repo_name, repo_info in config.items():
        db_path = os.path.join(repo_info['path'], f'{repo_name.lower().replace(" ", "_")}.db')
        if os.path.exists(db_path):
            print(f"Cleaning database for {repo_name}...")
            clean_titles(db_path)
        else:
            print(f"No database found for {repo_name}")

if __name__ == "__main__":
    main()