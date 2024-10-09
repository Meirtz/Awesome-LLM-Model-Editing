import argparse
import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.main import main

def parse_args():
    parser = argparse.ArgumentParser(description="Update arXiv paper repositories")
    parser.add_argument("--repos", nargs="+", default=["LLM-Paper-Daily"],
                        help="List of repositories to update (default: LLM-Paper-Daily)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.repos)