import os
import json
import time
from openai import OpenAI

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
                    answer = answer.replace('```json', '').replace('```', '').strip()
                    
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

    def generate_summary(self, title: str, abstract: str) -> str:
        prompt = f"Please provide a brief summary (2-3 sentences) of the following paper:\nTitle: {title}\nAbstract: {abstract}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes academic papers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()