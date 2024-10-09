import sqlite3
import os
from typing import List, Dict, Any

class PaperDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_dir_exists()
        self._create_table()

    def _ensure_dir_exists(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _create_table(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS papers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        abstract TEXT,
                        arxiv_id TEXT,
                        keywords TEXT,
                        category TEXT,
                        summary TEXT,
                        date_added DATE DEFAULT CURRENT_DATE
                    )
                ''')
        except sqlite3.OperationalError as e:
            print(f"Error creating table: {e}")
            print(f"Database path: {self.db_path}")
            print(f"Current working directory: {os.getcwd()}")
            raise

    def add_paper(self, paper: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO papers (title, abstract, arxiv_id, keywords, category, summary)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (paper['title'], paper['abstract'], paper.get('arxiv_id', ''), 
                  paper.get('keywords', ''), paper.get('category', ''), paper.get('summary', '')))

    def get_papers(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM papers ORDER BY date_added DESC')
            return [dict(row) for row in cursor.fetchall()]

    def paper_exists(self, title: str, arxiv_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM papers WHERE title = ? OR arxiv_id = ?', (title, arxiv_id))
            return cursor.fetchone() is not None