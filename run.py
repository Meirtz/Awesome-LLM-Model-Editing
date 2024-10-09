import sys
import os

# 将 src 目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import main

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the paper processing pipeline.")
    parser.add_argument('--repos', nargs='+', help='List of repositories to update', required=True)
    args = parser.parse_args()
    
    main(args.repos)