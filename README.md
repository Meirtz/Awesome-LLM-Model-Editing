# Auto-arXiv

Auto-arXiv is an automated research paper classification and translation project. Leveraging state-of-the-art language models and web scraping techniques, it systematically sorts and translates research papers from arXiv, making it easier for researchers and practitioners to stay updated with the latest advancements in the field of Large Language Models (LLMs). The project's core functionality is encapsulated in two primary classes: `LLMClassifier` and `PageParser`.

## Features

- **Automated Paper Classification**: Classify research papers based on their relevance to Large Language Models (LLMs) using OpenAI's language models.
- **Research Paper Extraction**: Extract research papers from arXiv's Computer Science section and organize them based on their titles and URLs.
- **Abstract Translation**: Translate the abstracts of relevant papers to different languages using the `googletrans` library, making them accessible to a global audience.
- **Retry Mechanism**: In-built retry mechanism to handle service unavailability issues while interacting with external APIs.
- **Export to File**: Save the classified and translated information in a text file for future reference or sharing.

## Setup

### Prerequisites

- Python 3.x
- Required libraries: `requests`, `beautifulsoup4`, `openai`, `googletrans`

```bash
pip install requests beautifulsoup4 openai googletrans
```

### Usage

1. Clone the repository:
```bash
git clone https://github.com/Meirtz/auto-arxiv.git
cd auto-arxiv
```
2. Export your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key'
```
3. Run the script:
```bash
python main.py
```
The script will extract research papers from the specified arXiv URL, classify them based on their relevance to LLMs, translate the abstracts, and save the information in a text file named `llm_related_papers_<today's-date>.txt`.

## Classes

### LLMClassifier

`LLMClassifier` is responsible for classifying papers and translating abstracts. It leverages OpenAI's language models for classification and `googletrans` for translation.

- **Methods**:
    - `__init__(self, api_key=None, model="gpt-4", max_tokens=50, temperature=0)`: Initializes the classifier.
    - `configure(self, model, max_tokens, temperature)`: Configures the classifier.
    - `classify(self, title, max_retries=3, retry_delay=5)`: Classifies a paper based on its title.
    - `parse(self, answer, max_retries=3, retry_delay=5)`: Parses the classification answer.
    - `translate(self, text, target_language='zh-cn')`: Translates text to a specified language.

### PageParser

`PageParser` is responsible for extracting papers and their abstracts from arXiv.

- **Methods**:
    - `extract_papers(self, url)`: Extracts paper information from a specified arXiv URL.
    - `extract_abstract(self, url)`: Extracts the abstract of a paper from a specified URL.

## Contributing

Feel free to fork the project, open issues, and submit pull requests. Your contributions are welcome!

---

Auto-arXiv is not affiliated with or endorsed by arXiv or OpenAI.
```