<<<<<<< HEAD
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
=======
# Awesome-LLM-Model-Editing

## Awesome LLM Model Editing

Welcome to the **Awesome LLM Model Editing** repository! This project curates a list of high-quality resources related to LLM Model Editing, including research papers, tools, libraries, and more.

## Daily Updates

### 2024-10-09

### 1. [Neurosymbolic AI approach to Attribution in Large Language Models](https://arxiv.org/pdf/2410.03726)

**Summary**: The paper proposes a Neurosymbolic AI (NesyAI) approach to improve attribution in large language models (LLMs), addressing issues like hallucinations, biases, and unreliable sources. By combining neural networks with structured symbolic reasoning, NesyAI aims to provide more reliable, interpretable, and adaptable systems for grounding LLM outputs with accurate and verifiable information.

### 2. [Revisiting the Superficial Alignment Hypothesis](https://arxiv.org/pdf/2410.03717)

**Summary**: The paper challenges the Superficial Alignment Hypothesis by demonstrating that post-training significantly enhances language models' performance on various tasks, scaling as a power law with the number of finetuning examples. It shows that post-training not only aligns models stylistically but also improves reasoning abilities, suggesting that the hypothesis oversimplifies the impact of post-training on model capabilities.

### 3. [Reasoning Elicitation in Language Models via Counterfactual Feedback](https://arxiv.org/pdf/2410.03767)

**Summary**: The paper introduces new metrics to evaluate the reasoning capabilities of language models, particularly in counterfactual scenarios, and proposes fine-tuning methods to improve these abilities. The study evaluates the fine-tuned models across various reasoning tasks, demonstrating enhanced generalization and performance compared to base models.

### 4. [Language Enhanced Model for Eye (LEME): An Open-Source Ophthalmology-Specific Large Language Model](https://arxiv.org/pdf/2410.03740)

**Summary**: The paper introduces LEME, an open-source ophthalmology-specific Large Language Model (LLM) that outperforms other LLMs in various validation tasks. LEME, built on the Llama2 70B framework and fine-tuned with a curated corpus of ophthalmology data, excels in abstract completion, fill-in-the-blank, multiple-choice questions, and clinical QA, demonstrating its potential to revolutionize clinical tasks and research collaboration in ophthalmology.

### 5. [Precision Knowledge Editing: Enhancing Safety in Large Language Models](https://arxiv.org/pdf/2410.03772)

**Summary**: The paper introduces Precision Knowledge Editing (PKE), a technique that enhances the safety of large language models (LLMs) by more effectively identifying and modifying toxic parameter regions. PKE, which builds on existing knowledge editing methods, uses neuron weight tracking and activation pathway tracing to achieve finer granularity in managing toxic content. Experiments show that PKE significantly reduces the attack success rate across various models while maintaining overall performance, outperforming closed-source models in terms of safety.

### 6. [Neuron-Level Sequential Editing for Large Language Models](https://arxiv.org/pdf/2410.04045)

**Summary**: The paper introduces Neuron-Level Sequential Editing (NSE), a novel method for continuously updating large language models (LLMs) through multi-round editing without requiring costly retraining. NSE optimizes hidden states using original weights to prevent model failure and iteratively selects neurons for editing to mitigate forgetting, outperforming existing methods in sequential model editing tasks.

### 7. [DiDOTS: Knowledge Distillation from Large-Language-Models for Dementia Obfuscation in Transcribed Speech](https://arxiv.org/pdf/2410.04188)

**Summary**: The paper introduces DiDOTS, a method that leverages Large-Language-Models (LLMs) to obfuscate dementia indicators in speech transcripts, addressing privacy concerns without relying on large labeled datasets. DiDOTS uses knowledge distillation to create a more efficient model with significantly fewer parameters, outperforming existing methods in both privacy and utility preservation.

### 8. [Persona Knowledge-Aligned Prompt Tuning Method for Online Debate](https://arxiv.org/pdf/2410.04239)

**Summary**: The paper introduces a novel framework that leverages ChatGPT's capabilities to simulate audience personas and align them with persona knowledge for assessing argument quality in online debates. By injecting audience persona knowledge into smaller language models through prompt tuning, the proposed method significantly improves performance over existing architectures, addressing the gap in combining argument persuasiveness with audience characteristics.

### 9. [Toxic Subword Pruning for Dialogue Response Generation on Large Language Models](https://arxiv.org/pdf/2410.04155)

**Summary**: The paper introduces ToxPrune, a novel algorithm that prunes toxic subwords from Byte Pair Encoding (BPE) in trained Large Language Models (LLMs) to prevent the generation of toxic content. Contrary to previous findings that pruning BPE tokens can harm machine translation tasks, the authors demonstrate that ToxPrune not only effectively reduces toxicity but also improves the performance of models like NSFW-3B and Llama-3.1-6B in dialogue response generation, enhancing both toxicity remediation and dialogue diversity.

### 10. [Mechanistic Behavior Editing of Language Models](https://arxiv.org/pdf/2410.04277)

**Summary**: The paper introduces TaRot, a method for task adaptation in large language models (LLMs) that uses learnable rotation matrices optimized via Bayesian Optimization to improve performance on classification and generation tasks. TaRot enhances both zero-shot and few-shot performance, with average improvements of 23.81% and 11.15% respectively across various models and tasks.

### 11. [Entity Insertion in Multilingual Linked Corpora: The Case of Wikipedia](https://arxiv.org/pdf/2410.04254)

**Summary**: The paper introduces the task of entity insertion in information networks, focusing on Wikipedia, and addresses the challenge of locating suitable positions for new links in text without predefined anchors. The authors develop a framework called LocEI and its multilingual variant XLocEI, which outperforms baseline models, including GPT-4, and demonstrates effective zero-shot performance across languages. This work is crucial for enhancing the linking process in Wikipedia's multilingual corpus.

### 12. [Wrong-of-Thought: An Integrated Reasoning Framework with Multi-Perspective Verification and Wrong Information](https://arxiv.org/pdf/2410.04463)

**Summary**: The paper introduces Wrong-of-Thought (WoT), an integrated reasoning framework designed to enhance the performance of Large Language Models (LLMs) by addressing two critical issues: the reliance on single verification methods and the ignorance of wrong information during reasoning. WoT incorporates multi-perspective verification and the utilization of wrong information to refine reasoning processes and reduce errors. Experimental results show that WoT outperforms previous methods across multiple datasets and LLMs, particularly in challenging computation tasks.

### 13. [Formality is Favored: Unraveling the Learning Preferences of Large Language Models on Data with Conflicting Knowledge](https://arxiv.org/pdf/2410.04784)

**Summary**: The study investigates how large language models (LLMs) handle conflicting information in training data, finding that they prefer formal texts and those with fewer spelling errors, leading to faster learning and better retention of knowledge. This preference is consistent across models and languages, with larger models showing a stronger inclination towards data that aligns with the majority of training data, suggesting that LLMs can be influenced by manipulating data consistency.

### 14. [Document-level Causal Relation Extraction with Knowledge-guided Binary Question Answering](https://arxiv.org/pdf/2410.04752)

**Summary**: The paper introduces a Knowledge-guided binary Question Answering (KnowQA) method for Event-Event Causal Relation Extraction (ECRE), addressing challenges like lack of document-level modeling and causal hallucinations. The proposed method, involving Event Structure Construction and Binary Question Answering, achieves state-of-the-art performance on the MECI dataset and demonstrates high generalizability and low inconsistency, especially with complete event structures post-fine-tuning.

### 15. [Activation Scaling for Steering and Interpreting Language Models](https://arxiv.org/pdf/2410.04962)

**Summary**: The paper explores the concept of steering language models by scaling activation vectors to correct incorrect predictions, such as flipping "Rome is in France" to "Rome is in Italy." The authors propose a three-term objective to ensure interventions are effective, faithful, and minimal. They demonstrate that activation scaling is more interpretable and efficient than steering vectors, allowing for precise model component identification and generalization across varying prompt lengths.

### 16. [ZEBRA: Zero-Shot Example-Based Retrieval Augmentation for Commonsense Question Answering](https://arxiv.org/pdf/2410.05077)

**Summary**: The paper introduces ZEBRA, a zero-shot question answering framework that enhances commonsense reasoning by combining retrieval, case-based reasoning, and introspection without requiring additional training of the language model. ZEBRA outperforms existing methods and strong language models, achieving an average accuracy improvement of up to 4.5 points across eight commonsense reasoning benchmarks.

### 17. [Deciphering the Interplay of Parametric and Non-parametric Memory in Retrieval-augmented Language Models](https://arxiv.org/pdf/2410.05162)

**Summary**: The study investigates how Retrieval-Augmented Generation (RAG) models like \textsc{Atlas} balance parametric (internal) and non-parametric (retrieved) memory during information processing. Through causal mediation analysis, it reveals that the model prioritizes retrieved context over internal knowledge when both are available, and identifies two key mechanisms: determining context relevance and computing output representations for copying relevant information.

### 18. [Metadata Matters for Time Series: Informative Forecasting with Transformers](https://arxiv.org/pdf/2410.03806)

**Summary**: The paper introduces MetaTST, a novel approach that integrates metadata into Transformer-based time series forecasting models to enhance accuracy and interpretability. By converting unstructured metadata into structured text and encoding it with large language models, MetaTST enriches the model's embedding with context-specific information, leading to improved performance across diverse forecasting scenarios. The method outperforms existing models in both short- and long-term forecasting benchmarks.

### 19. [A Pluggable Common Sense-Enhanced Framework for Knowledge Graph Completion](https://arxiv.org/pdf/2410.04488)

**Summary**: The paper introduces a pluggable framework for knowledge graph completion (KGC) that integrates both factual and common sense information to improve inference accuracy. The framework is adaptable to different KGs and includes mechanisms for generating common sense and optimizing negative sampling, enhancing performance across various KGC tasks. It can be integrated with existing knowledge graph embedding models, demonstrating scalability and superior performance compared to existing approaches.

### 20. [Augmenting Black-box LLMs with Medical Textbooks for Biomedical Question Answering (Published in Findings of EMNLP 2024)](https://arxiv.org/pdf/2309.02233)

**Summary**: The paper introduces LLM-AMT, a system that enhances large language models (LLMs) like ChatGPT with medical textbooks to improve their performance in biomedical question answering. By integrating medical textbooks through specialized modules, LLM-AMT significantly boosts accuracy in medical QA tasks, outperforming even specialized models like Med-PaLM 2. The study highlights that medical textbooks are more effective than Wikipedia as a knowledge source in this domain.

### 21. [Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue](https://arxiv.org/pdf/2401.04700)

**Summary**: The paper investigates the side effects of model editing on large language models (LLMs), finding that while editing improves factuality, it often degrades general abilities like reasoning and question answering. To address this, the authors propose RECT, a regularization method that constrains weight updates, effectively mitigating side effects while preserving editing performance.

### 22. [Corrective Retrieval Augmented Generation](https://arxiv.org/pdf/2401.15884)

**Summary**: The paper introduces Corrective Retrieval Augmented Generation (CRAG), a method designed to enhance the robustness of retrieval-augmented generation (RAG) by incorporating a retrieval evaluator and a decompose-then-recompose algorithm. CRAG aims to improve the quality of generated texts by assessing the relevance of retrieved documents and selectively focusing on key information, while also leveraging large-scale web searches to augment retrieval results. The approach is shown to significantly improve RAG performance across various datasets.

### 23. [Editing Conceptual Knowledge for Large Language Models](https://arxiv.org/pdf/2403.06259)

**Summary**: The paper introduces a novel approach to editing conceptual knowledge in Large Language Models (LLMs) by creating the ConceptEdit benchmark dataset and new evaluation metrics. It finds that while existing editing methods can modify concept-level definitions, they often distort related instantial knowledge, highlighting the need for improved techniques to balance these changes.

### 24. [FAC$^2$E: Better Understanding Large Language Model Capabilities by Dissociating Language and Cognition](https://arxiv.org/pdf/2403.00126)

**Summary**: The paper introduces FAC$^2$E, a framework for evaluating large language models (LLMs) by distinguishing between language and cognitive capabilities. It breaks down the evaluation process into three sub-steps: knowledge recall, knowledge utilization, and problem-solving, providing a detailed diagnosis of LLMs' performance. The study identifies a common weakness in knowledge utilization and suggests a knowledge-enhanced method to improve LLM performance, indicating potential future research directions.

### 25. [Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems](https://arxiv.org/pdf/2402.17840)

**Summary**: The paper investigates the vulnerability of Retrieval-Augmented Generation (RAG) systems, particularly those using instruction-tuned Language Models (LMs), to datastore leakage through prompt injection. The study reveals that adversaries can exploit these systems to extract verbatim text data from the datastore, with the risk increasing as model size scales up. The authors demonstrate successful extraction from various LMs and propose mitigation strategies, including position bias elimination, to reduce this vulnerability.

### 26. [Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete Knowledge Graph Question Answering](https://arxiv.org/pdf/2404.14741)

**Summary**: The paper introduces Generate-on-Graph (GoG), a training-free method that leverages Large Language Models (LLMs) to address Incomplete Knowledge Graph Question Answering (IKGQA) by generating new factual triples when the provided Knowledge Graph (KG) is insufficient. GoG operates through a Thinking-Searching-Generating framework, treating LLMs as both agents and KGs, and outperforms previous methods in experimental evaluations on two datasets.

### 27. [Student Data Paradox and Curious Case of Single Student-Tutor Model: Regressive Side Effects of Training LLMs for Personalized Learning](https://arxiv.org/pdf/2404.15156)

**Summary**: The paper identifies the "Student Data Paradox," where training Large Language Models (LLMs) on extensive student-tutor dialogue datasets to personalize education leads to a decline in the models' factual knowledge and reasoning abilities. The study demonstrates this paradox through quantitative analysis and introduces "hallucination tokens" as a partial solution, highlighting the ongoing challenge of balancing accurate student behavior modeling with maintaining the LLM's educational integrity.

### 28. [Red Teaming Language Models for Processing Contradictory Dialogues](https://arxiv.org/pdf/2405.10128)

**Summary**: The paper introduces a novel task for processing contradictory dialogues in language models, aiming to detect and modify self-contradictory statements. A Red Teaming framework is developed using a dataset of contradictory dialogues with explanatory labels, which improves detection, explanation, and modification of contradictions. The study underscores the significance of addressing logical inconsistencies in conversational AI.

### 29. [Exploring the Compositional Deficiency of Large Language Models in Mathematical Reasoning](https://arxiv.org/pdf/2405.06680)

**Summary**: The paper investigates the compositionality of large language models (LLMs) in mathematical reasoning by introducing logical traps into problem descriptions, creating a dataset called MathTrap. The study finds that while LLMs possess the necessary mathematical knowledge, they fail to spontaneously combine it with knowledge of logical traps to solve novel problems. Performance can be passively improved through external interventions like prompts and fine-tuning, but systematic compositionality remains a challenge for LLMs.

### 30. [Representation noising effectively prevents harmful fine-tuning on LLMs](https://arxiv.org/pdf/2405.14577)

**Summary**: The paper introduces Representation Noising (RepNoise), a defense mechanism against harmful fine-tuning attacks on large language models (LLMs). RepNoise removes information about harmful representations, making it difficult for attackers to recover them during fine-tuning, even when they have access to the model weights. The method is shown to be effective across different subsets of harm and does not impair the model's general capabilities or its ability to perform harmless tasks.

### 31. [WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models](https://arxiv.org/pdf/2405.14768)

**Summary**: The paper introduces WISE, a novel approach to lifelong model editing for large language models (LLMs), addressing the challenge of updating knowledge without compromising reliability, generalization, and locality. WISE employs a dual parametric memory scheme, with a main memory for pretrained knowledge and a side memory for edited knowledge, along with a router to manage queries. The proposed knowledge-sharding mechanism ensures that edits are stored in distinct subspaces, preventing conflicts and enabling effective merging into a shared memory. Experimental results demonstrate WISE's superior performance in various LLM architectures, overcoming the limitations of previous editing methods.

### 32. [RLSF: Reinforcement Learning via Symbolic Feedback](https://arxiv.org/pdf/2405.16661)

**Summary**: The paper introduces Reinforcement Learning via Symbolic Feedback (RLSF), a novel fine-tuning method for Large Language Models (LLMs) that leverages reasoning tools to provide detailed, token-level feedback through poly-sized certificates. This approach addresses the limitations of traditional RLHF methods by enhancing domain-specific understanding and enabling more effective fine-tuning. Evaluations demonstrate that RLSF outperforms traditional methods across various tasks, allowing smaller LLMs to surpass larger, closed-source models like GPT-4.

### 33. [Self-MoE: Towards Compositional Large Language Models with Self-Specialized Experts](https://arxiv.org/pdf/2406.12034)

**Summary**: The paper introduces Self-MoE, a method that converts monolithic large language models (LLMs) into modular systems of self-specialized experts (MiXSE) using self-generated synthetic data. This approach enhances the LLM's performance across various tasks without requiring extensive human-labeled data, showing significant improvements (6.5% on average) in benchmarks and outperforming other methods in flexibility and interpretability. The study underscores the importance of modularity and the potential for self-improvement in creating efficient and scalable LLM systems.

### 34. [FastMem: Fast Memorization of Prompt Improves Context Awareness of Large Language Models](https://arxiv.org/pdf/2406.16069)

**Summary**: The paper introduces FastMem, a method to improve the context awareness of large language models (LLMs) by quickly memorizing the prompt before inference. By optimizing only the last Feed-Forward Network module, FastMem enhances the model's ability to comprehend and follow context, leading to significant improvements in tasks like reading comprehension and text summarization. Experimental results show substantial gains in accuracy and adherence to output structures across different LLMs.

### 35. [DogeRM: Equipping Reward Models with Domain Knowledge through Model Merging](https://arxiv.org/pdf/2407.01470)

**Summary**: The paper introduces DogeRM, a framework that merges domain-specific knowledge into general reward models to improve their performance in reinforcement learning from human feedback (RLHF). By integrating expert annotations through model merging, DogeRM significantly enhances reward model accuracy across various benchmarks, demonstrating its potential to streamline the costly and time-consuming process of collecting preference data.

### 36. [To Forget or Not? Towards Practical Knowledge Unlearning for Large Language Models](https://arxiv.org/pdf/2407.01920)

**Summary**: The paper introduces KnowUnDo, a benchmark for evaluating knowledge unlearning in Large Language Models (LLMs), focusing on the inadvertent erasure of essential knowledge during the process of removing sensitive data. The authors propose MemFlex, a method that uses gradient information to precisely target and unlearn sensitive parameters, demonstrating superior performance in retaining general knowledge while effectively unlearning sensitive content.

### 37. [A Survey on Natural Language Counterfactual Generation](https://arxiv.org/pdf/2407.03993)

**Summary**: The paper surveys natural language counterfactual generation, which involves minimally altering texts to change their classification outcomes, thereby revealing model decision-making processes and improving robustness. It categorizes methods into four groups and reviews evaluation metrics, highlighting ongoing challenges and future research directions in leveraging Large Language Models for this task.

### 38. [Knowledge-based Consistency Testing of Large Language Models](https://arxiv.org/pdf/2407.12830)

**Summary**: The paper introduces KonTest, an automated framework for evaluating the consistency and knowledge gaps of Large Language Models (LLMs) using a knowledge graph. KonTest identifies inconsistencies and knowledge gaps in LLMs, with a 19.2% error rate and a 16.5% knowledge gap across tested models. The framework's mitigation method reduces knowledge gaps by 32.48%, and an ablation study highlights GPT3.5's limited effectiveness in knowledge construction.

### 39. [Knowledge Mechanisms in Large Language Models: A Survey and Perspective](https://arxiv.org/pdf/2407.15017)

**Summary**: The paper surveys knowledge mechanisms in Large Language Models (LLMs), categorizing them into knowledge utilization and evolution. It explores how LLMs memorize, comprehend, apply, and create knowledge, as well as how knowledge evolves within individual and group models. The study also addresses the fragility of parametric knowledge and hypothesizes about potential "dark knowledge" challenges, aiming to guide future research on understanding and enhancing LLMs.

### 40. [Optimal and efficient text counterfactuals using Graph Neural Networks](https://arxiv.org/pdf/2408.01969)

**Summary**: The paper introduces a framework for generating counterfactual explanations in NLP models using Graph Neural Networks, which creates semantically edited inputs that alter model predictions. The framework is tested on binary sentiment and topic classification tasks, demonstrating that it produces contrastive, fluent, and minimal edits while being significantly faster than existing methods.

### 41. [Revisiting Who's Harry Potter: Towards Targeted Unlearning from a Causal Intervention Perspective](https://arxiv.org/pdf/2407.16997)

**Summary**: The paper revisits the Who's Harry Potter (WHP) method for Large Language Model (LLM) unlearning, proposing a new task of targeted unlearning where only specific information about a given target is removed from the model. The authors introduce a causal intervention framework to model the unlearning process, treating the target's knowledge as a confounder and the unlearning as a deconfounding process. Their approach, which extends WHP, demonstrates competitive performance in experiments without explicit optimization for specific criteria.

### 42. [DYNAMICQA: Tracing Internal Knowledge Conflicts in Language Models](https://arxiv.org/pdf/2407.17023)

**Summary**: The paper introduces DynamicQA, a novel dataset designed to study intra-memory conflicts in Language Models (LMs), where conflicting knowledge within the model's parameters affects its ability to integrate relevant context. The dataset includes facts with temporal and disputable dynamics, allowing for the evaluation of semantic entropy and a new coherent persuasion score. The study finds that LMs exhibit more intra-memory conflict with dynamic facts and that these conflicts hinder the model's ability to update with new context, particularly in retrieval-augmented generation scenarios.

### 43. [Deconfounded Causality-aware Parameter-Efficient Fine-Tuning for Problem-Solving Improvement of LLMs](https://arxiv.org/pdf/2409.02686)

**Summary**: The paper investigates the reasoning limitations of Large Language Models (LLMs) and proposes a novel parameter-efficient fine-tuning method called Deconfounded Causal Adaptation (DCA) to enhance their problem-solving capabilities. By formulating the reasoning process into a causal framework and visualizing the text generation, the authors demonstrate that DCA significantly improves LLM performance across benchmarks with minimal tunable parameters, achieving better or comparable results to other fine-tuning methods.

### 44. [Beyond Persuasion: Towards Conversational Recommender System with Credible Explanations](https://arxiv.org/pdf/2409.14399)

**Summary**: The paper introduces PC-CRS, a method to enhance the credibility of explanations in conversational recommender systems (CRS) by guiding explanation generation with credibility-aware persuasive strategies and refining them through post-hoc self-reflection. Experimental results show that PC-CRS effectively promotes both persuasive and credible explanations, and further analysis suggests that credible explanations can improve recommendation accuracy.

### 45. [PEAR: Position-Embedding-Agnostic Attention Re-weighting Enhances Retrieval-Augmented Generation with Zero Inference Overhead](https://arxiv.org/pdf/2409.19745)

**Summary**: The paper introduces PEAR, a method that enhances the context awareness of large language models (LLMs) in retrieval-augmented generation (RAG) tasks without any inference overhead. PEAR identifies and re-weights attention heads that suppress context awareness, optimizing their impact through learnable coefficients. This approach improves RAG performance across various tasks while remaining agnostic to position embedding algorithms, offering both efficiency and broader applicability.

### 46. [Can Large Language Models Understand Symbolic Graphics Programs?](https://arxiv.org/pdf/2408.08313)

**Summary**: The paper investigates the ability of large language models (LLMs) to understand symbolic graphics programs, which require spatial-semantic reasoning without relying on vision encoders. The authors create a benchmark to evaluate LLMs' performance in this domain and introduce Symbolic Instruction Tuning (SIT) to enhance their understanding. The study finds that SIT not only improves performance on symbolic graphics tasks but also boosts general reasoning capabilities across various benchmarks.

### 47. [An Adversarial Perspective on Machine Unlearning for AI Safety](https://arxiv.org/pdf/2409.18025)

**Summary**: The paper examines the effectiveness of unlearning methods in removing hazardous capabilities from large language models, challenging their robustness against adversarial attacks. It demonstrates that existing jailbreak techniques can bypass unlearning protections and that adaptive methods can recover most unlearned capabilities, questioning the superiority of unlearning over traditional safety training.

### 48. [Representation Tuning](https://arxiv.org/pdf/2409.06927)

**Summary**: The paper introduces "representation tuning," a method for integrating behavioral vectors directly into large language models (LLMs) to enhance specific attributes like honesty without requiring online control. By fine-tuning the model with a dual loss function combining cosine similarity and token-based loss, the authors demonstrate that this approach outperforms traditional online steering and standard fine-tuning in improving model behavior, suggesting its potential as a safety measure.

### 49. [Adversarial Suffixes May Be Features Too!](https://arxiv.org/pdf/2410.00451)

**Summary**: The paper investigates the hypothesis that adversarial suffixes in large language models (LLMs) are not just vulnerabilities but may represent features that can dominate model behavior. Through experiments, the authors demonstrate that benign features can be transformed into adversarial suffixes that compromise safety alignment and that these features can be introduced through fine-tuning with benign datasets alone. The findings underscore the need for further research to enhance LLM safety alignment.



---

*Last updated on 2024-10-09*
>>>>>>> ebaa997b6d4c4f2c7ff446f030fd5e97dfe6ddb9
