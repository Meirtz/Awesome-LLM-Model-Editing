# Auto-Awesome

Auto-Awesome is an automated system for collecting, classifying, and summarizing research papers related to Large Language Models (LLMs) and other specified topics. It uses web scraping, machine learning classification, and database management to create curated lists of relevant papers.

## Features

- **Automated Paper Collection**: Scrapes recent papers from specified arXiv categories.
- **Intelligent Classification**: Uses a Large Language Model to classify papers based on relevance to specified topics.
- **Paper Summarization**: Generates concise summaries of relevant papers.
- **Database Management**: Stores and manages paper information in a SQLite database.
- **README Generation**: Automatically creates and updates README files for different paper collections.
- **Git Integration**: Automatically commits and pushes changes to specified Git repositories.

## Project Structure

- `src/`
  - `main.py`: Main script orchestrating the paper collection and processing pipeline.
  - `repo_manager.py`: Manages repository operations, including README updates and Git operations.
- `llm_classifier.py`: Handles paper classification and summarization using a Large Language Model.
- `paper_database.py`: Manages the SQLite database for storing paper information.
- `page_parser.py`: Extracts paper information from arXiv pages.
- `config.json`: Configuration file for repository settings.
- `prompts.json`: Contains prompt templates for paper classification.
- `readme_template.md`: Template for generating README files.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/Auto-Awesome.git
   cd Auto-Awesome
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY='your-api-key'
   ```

4. Configure the `config.json` and `prompts.json` files according to your needs.

## Usage

Run the main script with the repositories you want to update:

```
python run.py --repos Awesome-LLM
```