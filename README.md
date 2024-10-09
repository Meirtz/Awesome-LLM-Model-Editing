# Awesome-LLM-Model-Editing

## Awesome LLM Model Editing

Welcome to the **Awesome LLM Model Editing** repository! This project curates a list of high-quality resources related to LLM Model Editing, including research papers, tools, libraries, and more.

## Daily Updates

### 2024-10-09

### 1. [ERASMO: Leveraging Large Language Models for Enhanced Clustering Segmentation](https://arxiv.org/pdf/2410.03738)
**Summary**: The paper introduces ERASMO, a framework that leverages large language models to enhance clustering segmentation by fine-tuning on textually encoded tabular data. ERASMO transforms tabular data into a textual format, enabling the language model to generate contextually rich embeddings, which improve clustering accuracy by capturing complex relationships within multimodal datasets.

### 2. [Mitigating Training Imbalance in LLM Fine-Tuning via Selective Parameter Merging](https://arxiv.org/pdf/2410.03743)
**Summary**: The paper addresses the issue of training imbalance in Large Language Models (LLMs) during supervised fine-tuning (SFT) by proposing a method that merges models trained with different data orders. The novel "parameter-selection merging" technique is introduced, which outperforms traditional weighted-average methods and is validated through analysis and ablation studies on five datasets.

### 3. [Revisiting the Superficial Alignment Hypothesis](https://arxiv.org/pdf/2410.03717)
**Summary**: The paper challenges the Superficial Alignment Hypothesis by demonstrating that post-training with increasing fine-tuning examples significantly improves language model performance across various tasks, scaling similarly to pre-training. It highlights that post-training not only aligns style but also enhances reasoning abilities, suggesting that the hypothesis oversimplifies the impact of post-training on model capabilities.

### 4. [Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling](https://arxiv.org/pdf/2410.03735)
**Summary**: The paper introduces a method for creating task-specific language models by adjusting the training distribution of a generalist dataset using clustered importance sampling. This approach leverages the generalist data to simulate the specialist data, leading to improved performance in language modeling and multiple-choice tasks across various domains. The study also investigates the effects of dataset size, clustering configurations, and model size on the performance of the task-specific models.

### 5. [Progress Report: Towards European LLMs](https://arxiv.org/pdf/2410.03730)
**Summary**: The paper presents preliminary results from the OpenGPT-X project, which has developed two multilingual large language models (LLMs) capable of supporting all 24 official European Union languages. These models, trained on a dataset with 60% non-English content and optimized with a custom multilingual tokenizer, aim to overcome the limitations of existing LLMs that primarily focus on English. The models show competitive performance on various multilingual benchmarks, highlighting their effectiveness in addressing Europe's linguistic diversity.

### 6. [Beyond Scalar Reward Model: Learning Generative Judge from Preference Data](https://arxiv.org/pdf/2410.03742)
**Summary**: The paper proposes a novel approach to learning from preference data by using large language models (LLMs) to generate contrastive judgments with rationales, instead of relying on scalar reward models. This method, called Con-J, enhances interpretability and robustness against biases by directly optimizing the LLM's output based on preference data. Experimental results indicate that Con-J performs comparably to traditional scalar reward models while offering superior interpretability and bias resistance.

### 7. [Unsupervised Human Preference Learning](https://arxiv.org/pdf/2410.03731)
**Summary**: The paper introduces a novel method for personalized content generation by using small parameter models as preference agents to guide a larger pre-trained language model. This approach, which involves a "steering wheel" model directing the outputs of a foundation model, achieves efficient personalization without fine-tuning the large model. Experimental results show significant improvements over existing methods, enabling highly personalized language model applications.

### 8. [Language Enhanced Model for Eye (LEME): An Open-Source Ophthalmology-Specific Large Language Model](https://arxiv.org/pdf/2410.03740)
**Summary**: The paper introduces LEME, an open-source ophthalmology-specific Large Language Model (LLM) that outperforms other LLMs in various validation tasks. LEME, built on the Llama2 70B framework and fine-tuned with a large corpus of ophthalmology data, excels in abstract completion, fill-in-the-blank, multiple-choice questions, and clinical QA, demonstrating its potential to revolutionize clinical tasks and research collaboration in ophthalmology.

### 9. [Reasoning Elicitation in Language Models via Counterfactual Feedback](https://arxiv.org/pdf/2410.03767)
**Summary**: The paper introduces new metrics to evaluate the reasoning capabilities of language models, particularly in counterfactual scenarios, and proposes fine-tuning methods to enhance these abilities. The study evaluates the fine-tuned models across various reasoning tasks, demonstrating improved generalization and performance compared to base models.

### 10. [Khattat: Enhancing Readability and Concept Representation of Semantic Typography](https://arxiv.org/pdf/2410.03748)
**Summary**: The paper introduces Khattat, an end-to-end system for automating semantic typography, which enhances readability while visually representing word meanings. The system uses a Large Language Model to generate imagery ideas, FontCLIP to select appropriate fonts, and a diffusion model for iterative morphing, with an OCR-based loss function to maintain legibility. The method is shown to outperform baselines in readability and versatility across various languages and scripts.

### 11. [Precision Knowledge Editing: Enhancing Safety in Large Language Models](https://arxiv.org/pdf/2410.03772)
**Summary**: The paper introduces Precision Knowledge Editing (PKE), a technique that enhances the safety of large language models (LLMs) by more effectively identifying and modifying toxic parameter regions. PKE, which builds on existing knowledge editing methods, uses neuron weight tracking and activation pathway tracing to achieve finer granularity in managing toxic content. Experiments show that PKE significantly reduces the attack success rate across various models while maintaining overall performance, outperforming closed-source models in terms of safety.

### 12. [Reward-RAG: Enhancing RAG with Reward Driven Supervision](https://arxiv.org/pdf/2410.03780)
**Summary**: The paper introduces Reward-RAG, a method that enhances Retrieval-Augmented Generation (RAG) models by using a reward model trained with CriticGPT to fine-tune the RAG encoder, aligning its outputs more closely with human preferences. This approach shows significant performance improvements across various domains, demonstrating the effectiveness of integrating reward models with RAG for superior natural language generation.

### 13. [ORAssistant: A Custom RAG-based Conversational Assistant for OpenROAD](https://arxiv.org/pdf/2410.03845)
**Summary**: The paper introduces ORAssistant, a conversational assistant for the OpenROAD EDA tool, leveraging Retrieval-Augmented Generation (RAG) to enhance user assistance in tasks such as setup, decision-making, and flow automation. ORAssistant integrates multiple open-source tools and is designed to provide context-specific responses to user queries, demonstrating improved performance and accuracy over non-fine-tuned Large Language Models (LLMs).

### 14. [Self-Powered LLM Modality Expansion for Large Speech-Text Models](https://arxiv.org/pdf/2410.03798)
**Summary**: The paper introduces a self-powered approach to refine large speech-text models (LSMs) by addressing the issue of speech anchor bias, where models over-rely on speech inputs. By using augmented automatic speech recognition data generated by the model itself, the method effectively mitigates this bias and enhances the fusion of speech and text modalities, as demonstrated across various speech-based tasks.

### 15. [KidLM: Advancing Language Models for Children -- Early Insights and Future Directions](https://arxiv.org/pdf/2410.03884)
**Summary**: The paper introduces KidLM, a language model tailored for children, which focuses on using high-quality pre-training data and a novel training objective called Stratified Masking. This approach ensures the model understands lower grade-level text, maintains safety standards, and captures children's preferences, while also offering insights for future research in child-specific language modeling.

### 16. [Can Language Models Reason about Individualistic Human Values and Preferences?](https://arxiv.org/pdf/2410.03868)
**Summary**: The paper introduces IndieValueCatalog, a dataset derived from the World Values Survey, to assess language models' ability to reason about individualistic human values. The study finds that current models struggle with this task, achieving accuracies between 55% to 65%, and highlights the inadequacy of demographic information alone in predicting individual values. The authors propose the Value Inequity Index to measure biases and train Individualistic Value Reasoners to improve model performance, while outlining future research directions for individualistic alignment.

### 17. [Still Not Quite There! Evaluating Large Language Models for Comorbid Mental Health Diagnosis](https://arxiv.org/pdf/2410.03908)
**Summary**: The study introduces ANGST, a benchmark for multi-label classification of depression-anxiety comorbidity from social media posts, challenging models to identify both conditions simultaneously. Despite GPT-4's superior performance, none of the tested models exceeded an F1 score of 72%, highlighting the limitations of current language models in accurately diagnosing complex mental health conditions.

### 18. [ActPlan-1K: Benchmarking the Procedural Planning Ability of Visual Language Models in Household Activities](https://arxiv.org/pdf/2410.03907)
**Summary**: The paper introduces ActPlan-1K, a multi-modal planning benchmark designed to evaluate the procedural planning abilities of visual language models (VLMs) in household activities, considering both normal and counterfactual scenarios. The benchmark, constructed using ChatGPT and the iGibson2 simulator, includes 1,187 instances across 153 activities, each with natural language descriptions and environment images, and evaluates the correctness and commonsense satisfaction of generated plans. The study finds that current VLMs struggle to produce human-level plans and proposes automatic evaluation metrics using a fine-tuned BLEURT model to aid future research.

### 19. [Question-Answering System for Bangla: Fine-tuning BERT-Bangla for a Closed Domain](https://arxiv.org/pdf/2410.03923)
**Summary**: The paper introduces a question-answering system for Bengali using a fine-tuned BERT-Bangla model in a closed domain, specifically tailored for Khulna University of Engineering & Technology's content. Evaluated with 2500 question-answer pairs, the system achieved an Exact Match score of 55.26% and an F1 score of 74.21%, indicating potential for domain-specific applications but requiring further refinement for complex queries.

### 20. [Structured List-Grounded Question Answering](https://arxiv.org/pdf/2410.03950)
**Summary**: The paper introduces LIST2QA, a new dataset for evaluating question answering systems' ability to utilize structured list information, created from customer service documents using language models and filtering processes. The authors propose an Intermediate Steps for Lists (ISL) approach to enhance model performance, showing significant improvements in metrics like ROUGE-L, correctness, faithfulness, and completeness when compared to baseline models.

### 21. [Neuron-Level Sequential Editing for Large Language Models](https://arxiv.org/pdf/2410.04045)
**Summary**: The paper introduces Neuron-Level Sequential Editing (NSE), a novel method for continuously updating large language models (LLMs) through multi-round editing without requiring costly retraining. NSE optimizes hidden states using original weights to prevent model failure and iteratively selects neurons for editing to mitigate forgetting, outperforming existing methods in sequential model editing tasks.

### 22. [LoRTA: Low Rank Tensor Adaptation of Large Language Models](https://arxiv.org/pdf/2410.04060)
**Summary**: The paper introduces LoRTA, a novel method for Low Rank Tensor Adaptation of large language models, which significantly reduces the number of trainable parameters compared to traditional Low Rank Adaptation (LoRA). By employing low-rank tensor parametrization, LoRTA achieves efficient fine-tuning while maintaining performance across various benchmarks, demonstrating its effectiveness and efficiency in adapting large pre-trained models.

### 23. [Take It Easy: Label-Adaptive Self-Rationalization for Fact Verification and Explanation Generation](https://arxiv.org/pdf/2410.04002)
**Summary**: The paper introduces a label-adaptive self-rationalization method for fact verification and explanation generation, extending the approach from natural language inference tasks. By fine-tuning a model in two steps—first for veracity prediction and then for self-rationalization—the method significantly improves accuracy on PubHealth and AVeriTec datasets compared to GPT-4. Additionally, the use of synthetic explanations generated by large language models demonstrates the feasibility of low-cost learning, suggesting a promising direction for future explainable fact-checking research.

### 24. [Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks](https://arxiv.org/pdf/2410.04055)
**Summary**: The paper introduces a Self-Correction Learning (SCL) framework for Vision-Language Models (VLMs) to improve their reasoning abilities through self-generated self-correction data, collected during inference and fine-tuned using Direct Preference Optimization (DPO). The study shows that while VLMs struggle with self-correction during inference, preference fine-tuning significantly enhances their performance, suggesting that self-correction should be viewed as a learning process rather than mere refinement.

### 25. [PAD: Personalized Alignment at Decoding-Time](https://arxiv.org/pdf/2410.04070)
**Summary**: The paper introduces Personalized Alignment at Decoding-time (PAD), a novel framework that aligns Large Language Model (LLM) outputs with diverse personalized preferences during inference without requiring additional training. PAD uses a personalized reward modeling strategy to generate token-level rewards that guide the decoding process, enabling the model to adapt to various user preferences in real-time. Experimental results show that PAD outperforms traditional training-based methods and demonstrates generalizability and scalability across different base models.

### 26. [Exploring LLM-based Data Annotation Strategies for Medical Dialogue Preference Alignment](https://arxiv.org/pdf/2410.04112)
**Summary**: The paper explores using Reinforcement Learning from AI Feedback (RLAIF) to enhance healthcare dialogue models by addressing challenges in preference-aligned data annotation without relying on medical experts. It introduces a new evaluation framework based on standardized patient examinations and finds that an agent-based approach using Constitutional AI and flowcharts for expressing physician preferences outperforms existing methods, demonstrating strong generalization and reducing expert involvement.

### 27. [A Learning Rate Path Switching Training Paradigm for Version Updates of Large Language Models](https://arxiv.org/pdf/2410.04103)
**Summary**: The paper introduces a novel training paradigm for version updates of Large Language Models (LLMs) that combines the benefits of pre-training from scratch and continual pre-training. By strategically adjusting the learning rate during different stages of training, the proposed method achieves significant cost savings (reducing total training cost to 58% compared to pre-training from scratch) while maintaining competitive performance. This approach addresses the challenges of balancing performance and training cost in LLM version updates.

### 28. [Can the Variation of Model Weights be used as a Criterion for Self-Paced Multilingual NMT?](https://arxiv.org/pdf/2410.04147)
**Summary**: The paper introduces a novel algorithm for selecting minibatch languages in many-to-one neural machine translation, based on the variation of model weights measured by smoothed KL divergence. The algorithm outperforms alternating monolingual batches but does not surpass shuffled batches in terms of translation quality and convergence speed.

### 29. [DiDOTS: Knowledge Distillation from Large-Language-Models for Dementia Obfuscation in Transcribed Speech](https://arxiv.org/pdf/2410.04188)
**Summary**: The paper introduces DiDOTS, a method that leverages Large-Language-Models (LLMs) to obfuscate dementia indicators in speech transcripts, addressing privacy concerns without relying on large labeled datasets. DiDOTS uses knowledge distillation to create a more efficient model with significantly fewer parameters, achieving better privacy performance and utility preservation compared to existing methods.

### 30. [Toxic Subword Pruning for Dialogue Response Generation on Large Language Models](https://arxiv.org/pdf/2410.04155)
**Summary**: The paper introduces ToxPrune, a novel algorithm that prunes toxic subwords from Byte Pair Encoding (BPE) in trained Large Language Models (LLMs) to prevent the generation of toxic content. Contrary to previous findings that pruning BPE tokens can harm machine translation tasks, the authors demonstrate that ToxPrune not only effectively reduces toxicity but also improves the performance of models like NSFW-3B and Llama-3.1-6B in dialogue response generation, enhancing both toxicity reduction and dialogue diversity.

### 31. [Persona Knowledge-Aligned Prompt Tuning Method for Online Debate](https://arxiv.org/pdf/2410.04239)
**Summary**: The paper introduces a novel framework that leverages ChatGPT's capabilities to simulate audience personas and enhance argument quality assessment by aligning persona knowledge with smaller language models through prompt tuning. This approach significantly improves performance in online debate scenarios by integrating audience-specific characteristics into the evaluation process.

### 32. [Adaptive Question Answering: Enhancing Language Model Proficiency for Addressing Knowledge Conflicts with Source Citations](https://arxiv.org/pdf/2410.04241)
**Summary**: The paper introduces a novel framework for adaptive Question Answering (QA) that addresses knowledge conflicts by integrating source citations in ambiguous settings, where multiple valid answers exist. It presents five new datasets, a multi-hop QA dataset, two evaluation metrics, and several baselines to facilitate research in this area, aiming to enhance the trustworthiness and interpretability of QA systems.

### 33. [Consistent Autoformalization for Constructing Mathematical Libraries](https://arxiv.org/pdf/2410.04194)
**Summary**: The paper introduces a method to enhance autoformalization by combining three mechanisms: most-similar retrieval augmented generation (MS-RAG), denoising steps, and auto-correction with syntax error feedback (Auto-SEF). These mechanisms improve the consistency and reliability of translating natural language mathematical content into formal language expressions, particularly as the complexity of the domain increases. Empirical results show that these techniques enhance syntactic, terminological, and semantic consistency across various models.

### 34. [RoQLlama: A Lightweight Romanian Adapted Language Model](https://arxiv.org/pdf/2410.04269)
**Summary**: The paper introduces RoQLlama-7b, a quantized version of the Llama2 model adapted for Romanian tasks, demonstrating improved performance on seven Romanian downstream tasks in zero-shot and few-shot setups. Additionally, the authors contribute RoMedQA, a novel Romanian dataset for medical question answering.

### 35. [Evaluating Language Model Character Traits](https://arxiv.org/pdf/2410.04272)
**Summary**: The paper introduces a formal framework for evaluating character traits in language models, such as truthfulness and harmfulness, by examining consistent patterns of behavior. It finds that these traits can be stationary or reflective depending on the context and interaction, and their consistency varies with model size, fine-tuning, and prompting. The approach allows for precise characterization of LM behavior without anthropomorphism.

### 36. [Mechanistic Behavior Editing of Language Models](https://arxiv.org/pdf/2410.04277)
**Summary**: The paper introduces TaRot, a method for task adaptation in large language models (LLMs) that uses learnable rotation matrices optimized via Bayesian Optimization to improve performance on classification and generation tasks. TaRot enhances both zero-shot and few-shot performance, with average improvements of 23.81% and 11.15% respectively across various models and tasks.

### 37. [Correlation-Aware Select and Merge Attention for Efficient Fine-Tuning and Context Length Extension](https://arxiv.org/pdf/2410.04211)
**Summary**: The paper introduces a novel attention architecture that enables efficient fine-tuning and context length extension in large language models by using correlation-aware selection and merging mechanisms. It also proposes a data augmentation technique with positional encodings to enhance generalization. The method achieves significant resource savings and competitive performance, allowing models like Llama2-7B to handle context lengths up to 1M with reduced computational demands.

### 38. [ReTok: Replacing Tokenizer to Enhance Representation Efficiency in Large Language Model](https://arxiv.org/pdf/2410.04335)
**Summary**: The paper introduces ReTok, a method to enhance the efficiency of large language models by replacing their tokenizers. By reinitializing the input and output layers with the original model's parameters and training them while keeping other parameters fixed, the approach maintains model performance while significantly boosting decoding speed for long texts.

### 39. [Lens: Rethinking Multilingual Enhancement for Large Language Models](https://arxiv.org/pdf/2410.04407)
**Summary**: The paper introduces Lens, a novel method to enhance the multilingual capabilities of large language models (LLMs) by manipulating their internal language representation spaces. By drawing target languages closer to a central language in the language-agnostic subspace and pushing them apart in the language-specific subspace, Lens improves multilingual performance without compromising the original language capabilities of the model. This approach outperforms existing post-training methods with significantly reduced computational resources.

### 40. [Ordinal Preference Optimization: Aligning Human Preferences via NDCG](https://arxiv.org/pdf/2410.04346)
**Summary**: The paper introduces Ordinal Preference Optimization (OPO), a novel listwise approach that leverages the Normalized Discounted Cumulative Gain (NDCG) to better utilize ordinal rankings of multiple responses in aligning Large Language Models (LLMs) with human preferences. OPO outperforms existing pairwise and listwise methods in aligning multi-response datasets and demonstrates improved performance with an increased pool of negative samples.

### 41. [TIS-DPO: Token-level Importance Sampling for Direct Preference Optimization With Estimated Weights](https://arxiv.org/pdf/2410.04350)
**Summary**: The paper introduces TIS-DPO, a novel approach to Direct Preference Optimization (DPO) that addresses the issue of token importance in preference alignment for Large Language Models (LLMs). By using token-level importance sampling and estimating token weights through contrastive LLMs, TIS-DPO significantly improves performance on alignment tasks, outperforming existing methods. The study demonstrates the effectiveness of the approach through experiments and visualizations of token importance.

### 42. [DAdEE: Unsupervised Domain Adaptation in Early Exit PLMs](https://arxiv.org/pdf/2410.04424)
**Summary**: The paper introduces DAdEE, an unsupervised domain adaptation framework for Early Exit Pre-trained Language Models (PLMs) that addresses the issue of domain sensitivity in exit classifiers. By employing multi-level adaptation through knowledge distillation and GAN-based adversarial adaptation, DAdEE reduces the domain gap and enhances inference speed while improving domain adaptation performance across various tasks.

### 43. [Collapsed Language Models Promote Fairness](https://arxiv.org/pdf/2410.04472)
**Summary**: The paper investigates the phenomenon of Neural Collapse in language models to understand and improve fairness. It finds that debiased models exhibit collapsed alignment between token representations and word embeddings, which leads to the development of a fine-tuning method that enhances fairness across various debiasing techniques while maintaining model performance on standard tasks.

### 44. [Wrong-of-Thought: An Integrated Reasoning Framework with Multi-Perspective Verification and Wrong Information](https://arxiv.org/pdf/2410.04463)
**Summary**: The paper introduces Wrong-of-Thought (WoT), a novel reasoning framework designed to improve the performance of Large Language Models (LLMs) by addressing two key issues: the reliance on single verification methods and the ignorance of wrong information during reasoning. WoT incorporates multi-perspective verification to refine reasoning processes and utilizes wrong information to prevent recurring errors. Experimental results show that WoT outperforms existing methods across multiple datasets and LLMs, particularly in challenging computation tasks.

### 45. [Towards Secure Tuning: Mitigating Security Risks Arising from Benign Instruction Fine-Tuning](https://arxiv.org/pdf/2410.04524)
**Summary**: The paper investigates the security risks associated with Instruction Fine-Tuning (IFT) of Large Language Models (LLMs), even when using benign instructions. It introduces a novel Modular Layer-wise Learning Rate (ML-LR) strategy to mitigate these risks by analyzing module robustness and applying differentiated learning rates to robust modules. The study demonstrates that this approach significantly reduces the harmfulness of LLMs post-IFT without compromising their usability or expertise.

### 46. [DAMRO: Dive into the Attention Mechanism of LVLM to Reduce Object Hallucination](https://arxiv.org/pdf/2410.04514)
**Summary**: The paper introduces DAMRO, a training-free strategy to reduce object hallucination in Large Vision-Language Models (LVLMs) by analyzing and correcting the attention distribution of the LLM decoder. DAMRO uses the classification token (CLS) of ViT to filter out high-attention outlier tokens in the background, thereby improving the model's focus on relevant objects and reducing hallucination. Evaluations on multiple benchmarks show significant improvements in alleviating hallucination in LVLMs.

### 47. [Punctuation Prediction for Polish Texts using Transformers](https://arxiv.org/pdf/2410.04621)
**Summary**: The paper presents a solution for the Poleval 2022 Task 1, focusing on punctuation prediction for Polish texts. The approach employs a single HerBERT model fine-tuned on both competition data and an external dataset, achieving a Weighted F1 score of 71.44.

### 48. [Upsample or Upweight? Balanced Training on Heavily Imbalanced Datasets](https://arxiv.org/pdf/2410.04579)
**Summary**: The paper investigates the equivalence of upsampling (Temperature Sampling) and upweighting loss (Scalarization) in training language models on heavily imbalanced datasets, particularly in multilingual settings. It finds that while these methods are theoretically equivalent under full gradient descent, they diverge under stochastic gradient descent, with upsampling leading to faster convergence but higher risk of overfitting. The paper introduces Cooldown, a strategy that adjusts sampling temperature during training to balance convergence speed and overfitting, achieving competitive performance with computational efficiency.

### 49. [Leveraging Large Language Models for Suicide Detection on Social Media with Limited Labels](https://arxiv.org/pdf/2410.04501)
**Summary**: The paper introduces a novel approach for detecting suicidal content on social media using Large Language Models (LLMs), leveraging pseudo-labels generated by prompting LLMs and fine-tuning techniques. An ensemble model combining Qwen2-72B-Instruct, Llama3-8B, Llama3.1-8B, and Gemma2-9B significantly enhances detection accuracy, achieving F1 scores of 0.770 and 0.731 on public and private test sets, respectively. The study highlights the importance of model size in prompting performance, with larger LLMs yielding better results.

### 50. [LRQ-Fact: LLM-Generated Relevant Questions for Multimodal Fact-Checking](https://arxiv.org/pdf/2410.04616)
**Summary**: The paper introduces LRQ-Fact, an automated framework for multimodal fact-checking that uses Vision-Language Models and Large Language Models to generate relevant questions and answers for assessing the veracity of content. The framework includes a rule-based decision-maker to evaluate the generated information, and experiments demonstrate improved accuracy in detecting multimodal misinformation across different model backbones.

### 51. [Passage Retrieval of Polish Texts Using OKAPI BM25 and an Ensemble of Cross Encoders](https://arxiv.org/pdf/2410.04620)
**Summary**: The paper introduces a winning solution for the Poleval 2023 Task 3: Passage Retrieval challenge, which involves retrieving passages from Polish texts across trivia, legal, and customer support domains. The approach combines the OKAPI BM25 algorithm for document retrieval with an ensemble of multilingual Cross Encoders for reranking. Fine-tuning the reranker models improved performance in the trivia domain but led to worse results in the other domains.

### 52. [Control Large Language Models via Divide and Conquer](https://arxiv.org/pdf/2410.04628)
**Summary**: The paper examines the challenges of controlling large language models (LLMs) through prompt-based methods for Lexically Constrained Generation (LCG), identifying issues such as position bias, low responsiveness to decoding parameters, and difficulty with complex constraints. To address these limitations, the authors propose a Divide and Conquer Generation strategy, which significantly improves the success rate of LCG tasks, offering a promising approach for enhancing LLM performance in controlled text generation.

### 53. [Contrastive Learning to Improve Retrieval for Real-world Fact Checking](https://arxiv.org/pdf/2410.04657)
**Summary**: The paper introduces Contrastive Fact-Checking Reranker (CFR), a novel retriever designed to improve evidence retrieval for complex fact-checking tasks by leveraging a contrastive learning approach. CFR is fine-tuned using the AVeriTeC dataset, incorporating multiple training signals, and demonstrates a 6% improvement in veracity classification accuracy on the AVeriTeC dataset. The model's effectiveness is also validated across various other datasets, showing its potential for broader applicability in fact-checking scenarios.

### 54. [Rule-based Data Selection for Large Language Models](https://arxiv.org/pdf/2410.04715)
**Summary**: The paper introduces a novel rule-based framework for data selection in large language models (LLMs), leveraging the orthogonality of score vectors to evaluate and select rules. By using the determinantal point process (DPP) to identify independent rules, the method automates the selection of high-quality data for LLM training, outperforming conventional approaches in both rating precision and model performance across various tasks and domains.

### 55. [$\textbf{Only-IF}$:Revealing the Decisive Effect of Instruction Diversity on Generalization](https://arxiv.org/pdf/2410.04717)
**Summary**: The paper investigates the impact of instruction diversity on the generalization capabilities of large language models (LLMs). It finds that generalization only emerges when training data spans multiple semantic domains, and that cross-domain diversification, even with limited data, significantly enhances a model's adaptability. The study also shows that increasing the diversity of an established dataset, rather than just its size, is more effective for improving model performance, especially for specialist and generalist models.

### 56. [Efficient transformer with reinforced position embedding for language models](https://arxiv.org/pdf/2410.04731)
**Summary**: The paper introduces an efficient transformer architecture that enhances performance by reinforcing positional embedding, achieving superior results with fewer encoder-decoder layers. By concatenating positional encoding with trainable token embeddings and normalizing the token embedding matrix, the method significantly reduces training and validation losses, and training time, outperforming a baseline model in Portuguese-English translation tasks across multiple datasets.

### 57. [Formality is Favored: Unraveling the Learning Preferences of Large Language Models on Data with Conflicting Knowledge](https://arxiv.org/pdf/2410.04784)
**Summary**: The study investigates how large language models (LLMs) handle conflicting information in training data, finding that they prefer formal texts and those with fewer spelling errors, similar to human preferences. This preference leads to faster learning and better treatment of knowledge in data with these features, especially in larger models, and can be influenced by manipulating data consistency.

### 58. [Document-level Causal Relation Extraction with Knowledge-guided Binary Question Answering](https://arxiv.org/pdf/2410.04752)
**Summary**: The paper introduces a Knowledge-guided binary Question Answering (KnowQA) method for Event-Event Causal Relation Extraction (ECRE), addressing challenges like document-level modeling and causal hallucinations. The proposed method, involving Event Structure Construction and Binary Question Answering, achieves state-of-the-art performance on the MECI dataset and demonstrates high generalizability and low inconsistency, especially with complete event structures post-fine-tuning.

### 59. [Activation Scaling for Steering and Interpreting Language Models](https://arxiv.org/pdf/2410.04962)
**Summary**: The paper explores the concept of steering language models by scaling activation vectors to correct incorrect predictions, such as flipping "Rome is in France" to "Rome is in Italy." The authors propose a three-term objective to ensure interventions are effective, faithful, and minimal. They demonstrate that activation scaling is more interpretable and efficient than steering vectors, allowing for precise modifications to the model's internal workings.

### 60. [Intent Classification for Bank Chatbots through LLM Fine-Tuning](https://arxiv.org/pdf/2410.04925)
**Summary**: The study assesses the performance of large language models (LLMs) like SlovakBERT, Llama 8b instruct, and Gemma 7b instruct for intent classification in banking chatbots. Fine-tuned SlovakBERT demonstrated superior in-scope accuracy and lower out-of-scope false positive rates, making it the preferred model for this application.

### 61. [As Simple as Fine-tuning: LLM Alignment via Bidirectional Negative Feedback Loss](https://arxiv.org/pdf/2410.04834)
**Summary**: The paper introduces a novel loss function called Bidirectional Negative Feedback (BNF) for aligning large language models (LLMs), which addresses the instability and hyperparameter sensitivity issues of Direct Preference Optimization (DPO). BNF simplifies the alignment process to be as straightforward as supervised fine-tuning, without requiring extra hyperparameters or pairwise data. Experimental results show that BNF achieves strong performance on QA benchmarks while maintaining better reasoning abilities compared to other methods.

### 62. [Initialization of Large Language Models via Reparameterization to Mitigate Loss Spikes](https://arxiv.org/pdf/2410.05052)
**Summary**: The paper introduces a novel technique called weight scaling as reparameterization (WeSaR) to mitigate loss spikes in large language models by addressing the non-uniformity of parameter norms. WeSaR stabilizes training by uniformly scaling the norms of the original parameters through the introduction of gate parameters, leading to improved performance and faster convergence compared to other initialization methods.

### 63. [SparsePO: Controlling Preference Alignment of LLMs via Sparse Token Masks](https://arxiv.org/pdf/2410.05102)
**Summary**: The paper introduces SparsePO, a novel approach to preference optimization (PO) for language models that focuses on weighting tokens differently based on their relevance to human preferences. By learning sparse weight masks, SparsePO allows the model to prioritize specific tokens during training, leading to improved performance in tasks such as sentiment control, dialogue, summarization, and text-to-code generation. The method demonstrates superior results compared to existing PO methods, particularly in reasoning tasks.

### 64. [Explanation sensitivity to the randomness of large language models: the case of journalistic text classification](https://arxiv.org/pdf/2410.05085)
**Summary**: The paper investigates the impact of random elements in training large language models (LLMs) on the explainability of their predictions, using journalistic text classification in French as a case study. It finds that different random seeds yield models with similar accuracy but varying explanations, suggesting the need to characterize the statistical distribution of explanations. The study also explores a simpler model that offers stable explanations but lower accuracy, and demonstrates that incorporating features derived from LLM explanations can improve this simpler model.

### 65. [Enhancing Equity in Large Language Models for Medical Applications](https://arxiv.org/pdf/2410.05180)
**Summary**: The paper discusses the potential of large language models (LLMs) in medical applications but highlights significant inequities affecting specific racial, gender, and underrepresented groups. To address these disparities, the authors propose EquityGuard, a framework designed to detect and mitigate biases in LLM-based medical applications, thereby promoting equity across diverse populations.

### 66. [ReasoningRank: Teaching Student Models to Rank through Reasoning-Based Knowledge Distillation](https://arxiv.org/pdf/2410.05168)
**Summary**: The paper introduces ReasoningRank, a reranking approach that enhances transparency by generating explicit and comparison reasoning to explain document rankings. It uses large language models to create these explanations and distills the knowledge into smaller, more efficient student models, which improve reranking accuracy and provide interpretability without the computational burden of LLMs.

### 67. [Cookbook: A framework for improving LLM generative abilities via programmatic data generating templates](https://arxiv.org/pdf/2410.05224)
**Summary**: The paper introduces Cookbook, a framework for programmatically generating training data to improve large language models (LLMs) without relying on human or LLM-generated data, thereby avoiding privacy and legal issues. Cookbook uses simple patterns over random tokens to create datasets that enhance LLM performance on specific tasks, achieving up to a 52.7% accuracy improvement. The framework also optimizes multi-task performance by mixing data from various templates, leading to superior results on the GPT4ALL evaluation suite compared to other models.

### 68. [Causal Micro-Narratives](https://arxiv.org/pdf/2410.05252)
**Summary**: The paper introduces a method for classifying causal micro-narratives from text using only a subject-specific ontology of causes and effects. Applied to inflation narratives, the approach leverages a human-annotated dataset and evaluates several large language models, with the fine-tuned Llama 3.1 8B achieving high F1 scores. The study highlights linguistic ambiguity as a significant challenge and suggests the framework's potential for social science research.

### 69. [Differential Transformer](https://arxiv.org/pdf/2410.05258)
**Summary**: The paper introduces Diff Transformer, a novel architecture that enhances attention mechanisms by amplifying relevant context while reducing noise through a differential attention mechanism. This approach outperforms traditional Transformers in various language modeling tasks and offers significant improvements in practical applications like long-context modeling, key information retrieval, and hallucination mitigation. Diff Transformer also enhances robustness in in-context learning and reduces activation outliers, positioning it as a promising advancement in large language models.

### 70. [Data Advisor: Dynamic Data Curation for Safety Alignment of Large Language Models](https://arxiv.org/pdf/2410.05269)
**Summary**: The paper introduces Data Advisor, a method that enhances large language models (LLMs) by dynamically curating data to improve safety alignment. By monitoring and analyzing the generated data, Data Advisor identifies deficiencies and guides subsequent data generation to address these issues, thereby improving both data quality and coverage. Experiments show that Data Advisor effectively enhances model safety across multiple LLMs without compromising their utility.

### 71. [SFTMix: Elevating Language Model Instruction Tuning with Mixup Recipe](https://arxiv.org/pdf/2410.05248)
**Summary**: The paper introduces SFTMix, a novel approach to improve instruction-tuning in large language models (LLMs) by leveraging Mixup-based regularization to address uneven confidence levels in the semantic representation space. This method enhances performance across various tasks without requiring high-quality curated datasets, demonstrating scalability and adaptability across different LLM families and datasets.

### 72. [SQFT: Low-cost Model Adaptation in Low-precision Sparse Foundation Models](https://arxiv.org/pdf/2410.03750)
**Summary**: The paper introduces SQFT, a method for low-precision sparse parameter-efficient fine-tuning of large pre-trained models (LPMs) in resource-constrained environments. SQFT enables the merging of sparse weights with low-rank adapters while maintaining sparsity and accuracy, addressing challenges related to different numerical precisions. The approach is validated across various adaptation scenarios and sparsity levels, demonstrating its effectiveness.

### 73. [Metadata Matters for Time Series: Informative Forecasting with Transformers](https://arxiv.org/pdf/2410.03806)
**Summary**: The paper introduces MetaTST, a novel approach that integrates metadata into Transformer-based time series forecasting models to enhance accuracy and interpretability. By converting unstructured metadata into structured text and encoding it with large language models, MetaTST enriches the model's embedding with contextual information, leading to improved performance across various forecasting scenarios. The method outperforms existing models in both short- and long-term forecasting benchmarks.

### 74. [Hyperbolic Fine-tuning for Large Language Models](https://arxiv.org/pdf/2410.04010)
**Summary**: The paper investigates the suitability of Euclidean space for embedding tokens in large language models (LLMs) and finds that token embeddings exhibit a high degree of hyperbolicity, suggesting a tree-like structure. To leverage this, the authors propose HypLoRA, a method for fine-tuning LLMs in hyperbolic space, which significantly improves performance on complex reasoning tasks, as demonstrated by a 13.0% improvement on the AQuA dataset.

### 75. [Learning Code Preference via Synthetic Evolution](https://arxiv.org/pdf/2410.03837)
**Summary**: The paper introduces CodeFavor, a framework for training pairwise code preference models using synthetic evolution data, and CodePrefBench, a benchmark for evaluating code preferences across correctness, efficiency, and security. The evaluation demonstrates that CodeFavor significantly improves model accuracy and cost-effectiveness compared to larger models, while highlighting the limitations and costs of human-based code preference assessments.

### 76. [DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search](https://arxiv.org/pdf/2410.03864)
**Summary**: The paper introduces DOTS, a method that allows large language models (LLMs) to dynamically reason by searching for optimal reasoning trajectories tailored to each question and the LLM's capabilities. The approach involves defining atomic reasoning actions, searching for the best action sequences for training questions, and using these sequences to train the LLM to plan reasoning for new questions. Experiments demonstrate that DOTS outperforms static reasoning methods and improves LLMs' ability to adapt reasoning depth based on problem complexity.

### 77. [Improving Arabic Multi-Label Emotion Classification using Stacked Embeddings and Hybrid Loss Function](https://arxiv.org/pdf/2410.03979)
**Summary**: The paper introduces a novel approach to improve Arabic multi-label emotion classification by combining stacked embeddings from fine-tuned language models (ArabicBERT, MarBERT, and AraBERT), a meta-learner, and a hybrid loss function that addresses class imbalance and label correlation. The proposed model significantly enhances performance across various metrics, demonstrating its effectiveness in balancing emotion classification for minority classes and offering a generalizable framework for other languages and domains.

### 78. [Realizing Video Summarization from the Path of Language-based Semantic Understanding](https://arxiv.org/pdf/2410.04511)
**Summary**: The paper introduces a novel video summarization framework that leverages the strengths of multiple Video-based Large Language Models (VideoLLMs) without requiring fine-tuning, inspired by the Mixture of Experts (MoE) paradigm. This approach generates comprehensive and coherent textual summaries by integrating visual and audio content, enhancing semantic understanding and performance in downstream tasks like summary video generation.

### 79. [Deeper Insights Without Updates: The Power of In-Context Learning Over Fine-Tuning](https://arxiv.org/pdf/2410.04691)
**Summary**: The paper challenges the conventional belief that fine-tuning outperforms in-context learning (ICL) with sufficient training data, demonstrating that ICL excels at capturing implicit patterns in tasks. Through experiments on specialized datasets, ICL models showed superior accuracy and pattern understanding compared to fine-tuned models, even when the latter used vastly more training samples. The authors propose a circuit shift theory to explain ICL's effectiveness.

### 80. [TLDR: Token-Level Detective Reward Model for Large Vision Language Models](https://arxiv.org/pdf/2410.04734)
**Summary**: The paper introduces TLDR, a Token-Level Detective Reward Model designed to provide fine-grained annotations for each text token in multimodal language models, addressing the limitations of existing binary feedback systems. TLDR uses a perturbation-based method to generate synthetic hard negatives and their token-level labels, enhancing model performance and speeding up human annotation processes.

### 81. [ImProver: Agent-Based Automated Proof Optimization](https://arxiv.org/pdf/2410.04753)
**Summary**: The paper introduces ImProver, an agent-based system using large language models (LLMs) to optimize formal proofs in Lean by rewriting them to meet user-defined criteria such as length, readability, or modularity. ImProver incorporates improvements like the Chain-of-States technique and error-correction mechanisms, demonstrating its ability to significantly enhance the quality of proofs across various mathematical domains.

### 82. [Regressing the Relative Future: Efficient Policy Optimization for Multi-turn RLHF](https://arxiv.org/pdf/2410.04612)
**Summary**: The paper introduces REFUEL, an efficient policy optimization approach for multi-turn reinforcement learning from human feedback (RLHF) in large language models (LLMs). REFUEL addresses the covariate shift issue by using a single model to estimate Q-values and training on self-generated data, framing the problem as a sequence of regression tasks. Empirical results show that REFUEL outperforms state-of-the-art methods and enables smaller models to outperform larger ones in long multi-turn dialogues.

### 83. [Can LLMs plan paths with extra hints from solvers?](https://arxiv.org/pdf/2410.05045)
**Summary**: The paper investigates how Large Language Models (LLMs) can be enhanced in solving robotic planning tasks by incorporating feedback from solvers. It explores various feedback strategies and evaluates the performance of three LLMs on a range of planning problems. The study finds that solver-generated feedback improves LLM performance on moderately difficult problems but struggles with more complex ones, highlighting the models' limitations in long-term planning and higher-order reasoning.

### 84. [DEPT: Decoupled Embeddings for Pre-training Language Models](https://arxiv.org/pdf/2410.05021)
**Summary**: The paper introduces DEPT, a novel pre-training framework that decouples embedding layers from the transformer model, allowing for more efficient and effective training across heterogeneous data sources. DEPT reduces parameter count and communication costs, enhances model generalization, and enables custom vocabularies per data source, demonstrated through a 1.3 billion-parameter model pre-training across diverse languages.

### 85. [Preserving Multi-Modal Capabilities of Pre-trained VLMs for Improving Vision-Linguistic Compositionality](https://arxiv.org/pdf/2410.05210)
**Summary**: The paper introduces Fine-grained Selective Calibrated CLIP (FSC-CLIP) to enhance compositional understanding in pre-trained vision and language models without degrading multi-modal performance. By integrating local hard negative loss and selective calibrated regularization, FSC-CLIP maintains fine-grained negative supervision while preserving representational integrity, achieving state-of-the-art compositionality and strong multi-modal capabilities across benchmarks.

### 86. [Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective](https://arxiv.org/pdf/2410.05192)
**Summary**: The paper introduces the Warmup-Stable-Decay (WSD) learning rate schedule, which allows for indefinite training without a fixed compute budget by maintaining a constant learning rate and then rapidly decaying it when needed. The authors propose a "river valley" loss landscape to explain the observed behavior, where a high stable learning rate drives rapid progress along the river, while a fast decay phase moves the model closer to the river's edge, revealing true optimization progress. This approach outperforms traditional methods in obtaining strong language model checkpoints across various compute budgets.

### 87. [Diversity Over Size: On the Effect of Sample and Topic Sizes for Topic-Dependent Argument Mining Datasets](https://arxiv.org/pdf/2205.11472)
**Summary**: The paper investigates the impact of dataset composition on Argument Mining performance in few- and zero-shot settings, finding that carefully selected training samples can significantly reduce the required dataset size while maintaining high performance. The study demonstrates that even with a 90% reduction in sample size, models can achieve 95% of maximum performance across multiple Argument Mining tasks and datasets.

### 88. [Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue](https://arxiv.org/pdf/2401.04700)
**Summary**: The paper investigates the trade-offs between improving the factuality of large language models (LLMs) through model editing and maintaining their general abilities in reasoning and other tasks. It finds that current editing methods often degrade the model's overall performance due to excessive weight alterations. To address this, the authors propose a regularization method called RECT, which constrains the complexity of weight updates, effectively mitigating side effects while preserving high editing performance.

### 89. [PILLOW: Enhancing Efficient Instruction Fine-tuning via Prompt Matching](https://arxiv.org/pdf/2312.05621)
**Summary**: The paper introduces PILLOW, a method to enhance the performance of Low-Rank Adaptation (LoRA) in fine-tuning Large Language Models (LLMs) by using a discrimination-based prompting approach. PILLOW leverages in-context learning and a matching network to select prompts from a user-defined pool, significantly reducing computational costs while maintaining comparable performance to traditional instruction fine-tuning methods, even on consumer-grade GPUs.

### 90. [R-Judge: Benchmarking Safety Risk Awareness for LLM Agents](https://arxiv.org/pdf/2401.10019)
**Summary**: The paper introduces R-Judge, a benchmark designed to assess the safety risk awareness of large language model (LLM) agents in interactive environments. It includes 569 multi-turn interaction records across 27 risk scenarios and evaluates 11 LLMs, showing that even the best-performing model, GPT-4o, achieves only 74.42% accuracy, indicating significant room for improvement. The study highlights the complexity of risk awareness in open scenarios and suggests that fine-tuning on safety judgment is more effective than simple prompting.

### 91. [Prompt-Based Bias Calibration for Better Zero/Few-Shot Learning of Language Models](https://arxiv.org/pdf/2402.10353)
**Summary**: The paper introduces a null-input prompting method to calibrate intrinsic bias in pre-trained language models, aiming to improve zero/few-shot learning performance while maintaining computational efficiency. By using GPT-4-generated null-meaning inputs and a distribution disparity loss, the method adjusts bias parameters to create a more equitable probability distribution, leading to significant enhancements in zero/few-shot learning across various datasets.

### 92. [Pedagogical Alignment of Large Language Models](https://arxiv.org/pdf/2402.05000)
**Summary**: The paper explores the use of Learning from Human Preferences (LHP) algorithms to align Large Language Models (LLMs) with effective teaching strategies, termed "pedagogical alignment." By generating synthetic datasets to overcome the scarcity of high-quality preference data, the study demonstrates that LHP methods outperform standard supervised fine-tuning, significantly improving the models' ability to guide students through problem-solving rather than providing direct answers. The authors also introduce new perplexity-based metrics to evaluate pedagogical alignment, highlighting the potential of LHP methods in enhancing LLMs' educational effectiveness.

### 93. [Unraveling Babel: Exploring Multilingual Activation Patterns of LLMs and Their Applications](https://arxiv.org/pdf/2402.16367)
**Summary**: The paper investigates the internal neuron activities of large language models (LLMs) when processing different languages by converting dense models into fine-grained Mixture of Experts (MoE) architectures. Through visual analysis and experiments, it uncovers patterns in expert activation frequencies and their implications for multilingual processing. The study demonstrates that leveraging these patterns can enhance sparse activation and pruning techniques, leading to improved model performance and efficiency.

### 94. [Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning](https://arxiv.org/pdf/2402.13950)
**Summary**: The paper investigates the faithfulness of reasoning steps in large language models (LLMs) and introduces FRODO, a framework that enhances the reliability of intermediate reasoning steps and their influence on final answers. FRODO outperforms existing methods by improving robustness and generalization, ensuring that the model's final predictions are more consistent with its stated reasoning.

### 95. [Editing Conceptual Knowledge for Large Language Models](https://arxiv.org/pdf/2403.06259)
**Summary**: The paper introduces a novel approach to editing conceptual knowledge in Large Language Models (LLMs) by creating the ConceptEdit benchmark dataset and new evaluation metrics. It finds that while existing editing methods can modify concept-level definitions, they often distort related instantial knowledge, highlighting the need for improved techniques to balance these changes.

### 96. [Head-wise Shareable Attention for Large Language Models](https://arxiv.org/pdf/2402.11819)
**Summary**: The paper introduces head-wise shareable attention as a method to reduce the memory footprint of Large Language Models (LLMs) by sharing parameters across attention heads. It proposes two techniques, **DirectShare** and **PostShare**, which either reuse pre-trained weights directly or post-train with constraints on weight similarity before sharing. The study demonstrates that fine-grained weight sharing maintains model performance, making it feasible for deployment on edge devices.

### 97. [HateCOT: An Explanation-Enhanced Dataset for Generalizable Offensive Speech Detection via Large Language Models](https://arxiv.org/pdf/2403.11456)
**Summary**: The paper introduces HateCOT, a large English dataset with over 52,000 samples, enhanced with GPT-3.5Turbo-generated explanations, aimed at improving the generalization of offensive speech detection models. Pretraining on HateCOT significantly boosts the performance of Large Language Models on benchmark datasets, even in zero-shot and few-shot scenarios, while also enhancing the quality of model explanations.

### 98. [sDPO: Don't Use Your Data All at Once](https://arxiv.org/pdf/2403.19270)
**Summary**: The paper introduces sDPO, an extension of direct preference optimization (DPO) that aligns large language models (LLMs) with human preferences by using preference datasets in a stepwise manner. This approach allows for more precise alignment and results in a final model that outperforms other LLMs, including those with more parameters.

### 99. [MetaAligner: Towards Generalizable Multi-Objective Alignment of Language Models](https://arxiv.org/pdf/2403.17141)
**Summary**: The paper introduces MetaAligner, a novel approach for multi-objective preference alignment in large language models that is both policy-agnostic and generalizable. It achieves this through a three-stage process: dynamic objectives reformulation, conditional weak-to-strong correction, and a generalizable inference method. MetaAligner significantly reduces training costs and can align with unseen objectives, outperforming existing methods in multi-objective alignment across various policy models.

### 100. [SpaceByte: Towards Deleting Tokenization from Large Language Modeling](https://arxiv.org/pdf/2404.14408)
**Summary**: The paper introduces SpaceByte, a byte-level decoder architecture designed to eliminate the need for tokenization in large language models while maintaining performance. By incorporating larger transformer blocks after specific byte sequences like spaces, SpaceByte significantly improves byte-level modeling and achieves performance comparable to tokenized models, demonstrating its effectiveness within a fixed computational budget.

### 101. [Red Teaming Language Models for Processing Contradictory Dialogues](https://arxiv.org/pdf/2405.10128)
**Summary**: The paper introduces a novel task for processing contradictory dialogues in language models, aiming to detect and modify self-contradictory statements. A Red Teaming framework is developed using a dataset of contradictory dialogues with explanatory labels, which improves detection, explanation, and modification of contradictions. The study underscores the significance of addressing logical inconsistencies in conversational AI.

### 102. [Representation noising effectively prevents harmful fine-tuning on LLMs](https://arxiv.org/pdf/2405.14577)
**Summary**: The paper introduces Representation Noising (RepNoise), a defense mechanism against harmful fine-tuning attacks on large language models (LLMs). RepNoise removes information related to harmful representations, making it difficult for attackers to recover them during fine-tuning, even when they have access to the model weights. The method is shown to be effective across various harmful tasks and does not impair the model's performance on harmless tasks.

### 103. [Promoting Constructive Deliberation: Reframing for Receptiveness](https://arxiv.org/pdf/2405.15067)
**Summary**: The paper introduces a method to enhance online discussions by automatically reframing disagreeing responses to appear more receptive, based on six identified strategies. Experiments using a Reddit dataset show that these reframed replies are perceived as significantly more receptive than original replies, demonstrating the potential of computational frameworks to align language models with human social constructs for improved content moderation.

### 104. [Exploring the Compositional Deficiency of Large Language Models in Mathematical Reasoning](https://arxiv.org/pdf/2405.06680)
**Summary**: The paper investigates the compositionality of large language models (LLMs) in mathematical reasoning by introducing logical traps into problem descriptions, creating a dataset called MathTrap. The study finds that while LLMs possess the necessary mathematical knowledge, they fail to spontaneously combine it with knowledge of logical traps to solve novel cases. Performance can be passively improved through external interventions like prompts and fine-tuning, but systematic compositionality remains a challenge for LLMs.

### 105. [WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models](https://arxiv.org/pdf/2405.14768)
**Summary**: The paper introduces WISE, a novel approach to lifelong model editing for large language models (LLMs), addressing the challenge of updating knowledge without compromising reliability, generalization, and locality. WISE employs a dual parametric memory scheme, distinguishing between pretrained knowledge and edited knowledge, and uses a router to manage queries. It also features a knowledge-sharding mechanism to prevent conflicts during continual editing, demonstrating superior performance in various LLM architectures.

### 106. [Student Data Paradox and Curious Case of Single Student-Tutor Model: Regressive Side Effects of Training LLMs for Personalized Learning](https://arxiv.org/pdf/2404.15156)
**Summary**: The paper identifies the "Student Data Paradox," where training Large Language Models (LLMs) on extensive student-tutor dialogue datasets to personalize education leads to a decline in the models' factual knowledge and reasoning abilities. The study demonstrates this paradox through quantitative analysis and introduces "hallucination tokens" as a partial solution, highlighting the ongoing challenge of balancing accurate student behavior modeling with maintaining the LLM's educational integrity.

### 107. [RLSF: Reinforcement Learning via Symbolic Feedback](https://arxiv.org/pdf/2405.16661)
**Summary**: The paper introduces Reinforcement Learning via Symbolic Feedback (RLSF), a novel approach to fine-tuning Large Language Models (LLMs) that leverages reasoning tools to provide detailed, token-level feedback, overcoming limitations of traditional reward models. RLSF enables smaller LLMs to outperform larger models like GPT-4 across various tasks, demonstrating its effectiveness in enhancing domain-specific understanding.

### 108. [Robo-Instruct: Simulator-Augmented Instruction Alignment For Finetuning CodeLLMs](https://arxiv.org/pdf/2405.20179)
**Summary**: The paper introduces ROBO-INSTRUCT, a method that combines a robot simulator (ROBOSIM) with an instruction-program alignment process (INSTALIGN) to enhance the fine-tuning of Code LLMs for domain-specific robot applications. By dynamically synthesizing simulation environments and aligning instructions with generated programs, ROBO-INSTRUCT improves the accuracy and diversity of training data, leading to a significant performance boost in fine-tuned models, outperforming some proprietary LLMs.

### 109. [ComplexTempQA: A Large-Scale Dataset for Complex Temporal Question Answering](https://arxiv.org/pdf/2406.04866)
**Summary**: The paper introduces ComplexTempQA, a massive dataset with over 100 million question-answer pairs, designed to challenge AI models in temporal question answering. It surpasses existing benchmarks in scale and complexity, featuring questions that require sophisticated reasoning across time, entities, and events, and includes detailed metadata for comprehensive evaluation.

### 110. [Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization](https://arxiv.org/pdf/2406.01171)
**Summary**: The paper surveys the emerging field of persona-based applications in large language models (LLMs), categorizing current research into two main areas: LLM Role-Playing, where models are assigned specific personas, and LLM Personalization, where models adapt to user personas. It introduces a systematic taxonomy and methods for evaluating LLM personality, aiming to unify and advance the understanding of persona in LLMs.

### 111. [Pcc-tuning: Breaking the Contrastive Learning Ceiling in Semantic Textual Similarity](https://arxiv.org/pdf/2406.09790)
**Summary**: The paper investigates the limitations of contrastive learning in achieving higher semantic textual similarity (STS) scores, identifying an upper limit of 87.5 for Spearman's correlation under this approach. To overcome this ceiling, the authors introduce Pcc-tuning, which uses Pearson's correlation coefficient as a loss function, significantly improving STS performance with minimal fine-grained annotations.

### 112. [Learn Beyond The Answer: Training Language Models with Reflection for Mathematical Reasoning](https://arxiv.org/pdf/2406.12050)
**Summary**: The paper introduces "reflective augmentation," a novel technique that enhances language models' mathematical reasoning by embedding problem reflection into training instances. This method encourages the model to consider alternative perspectives and engage with abstractions, improving performance in both standard and complex reflective reasoning tasks. The approach is shown to be effective and complementary to existing data augmentation techniques.

### 113. [Understanding Jailbreak Success: A Study of Latent Space Dynamics in Large Language Models](https://arxiv.org/pdf/2406.09289)
**Summary**: The paper investigates the mechanisms behind jailbreaking in large language models, finding that a single jailbreak vector can mitigate different types of jailbreaks, suggesting a common internal mechanism. It also identifies a potential commonality in how effective jailbreaks reduce the model's perception of prompt harmfulness, providing insights for developing more robust countermeasures against jailbreaking.

### 114. [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/pdf/2406.08464)
**Summary**: The paper introduces Magpie, a method for synthesizing high-quality instruction data by prompting aligned large language models (LLMs) with partial templates, enabling the generation of millions of instructions and responses. The synthesized data, when used for fine-tuning, demonstrates performance comparable to models trained on private, high-cost datasets, and outperforms previous public datasets on alignment benchmarks.

### 115. [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/pdf/2406.11695)
**Summary**: The paper introduces MIPRO, an algorithm for optimizing prompts in multi-stage Language Model Programs (LMPs) by refining instructions and demonstrations without module-level labels or gradients. MIPRO employs program- and data-aware techniques, a stochastic evaluation function, and a meta-optimization procedure to enhance performance, achieving up to 13% higher accuracy than baseline optimizers on diverse LMP tasks using the Llama-3-8B model.

### 116. [Test-Time Fairness and Robustness in Large Language Models](https://arxiv.org/pdf/2406.07685)
**Summary**: The paper introduces a novel approach to address biases in Large Language Models (LLMs) at test time, particularly when dealing with spurious features in inputs. Instead of relying on the model's implicit understanding of bias, the authors propose a stratified invariance framework based on causality, which allows for explicit debiasing requirements from population to individual levels. They demonstrate that their prompting strategy effectively reduces bias in LLMs across various benchmarks without additional data or retraining.

### 117. [Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models](https://arxiv.org/pdf/2406.08811)
**Summary**: The paper introduces Mixture-of-Skills (MoS), a reinforcement learning framework designed to optimize data usage during the fine-tuning of large language models (LLMs), addressing the challenges posed by heterogeneous and imbalanced datasets. MoS dynamically adjusts the focus on different datasets to ensure comprehensive skill development, and its effectiveness is demonstrated through experiments with diverse LLM backbones and benchmarks. Additionally, the paper proposes MoSpec, an adaptation for task-specific fine-tuning, highlighting the importance of dataset rebalancing in LLM fine-tuning.

### 118. [COMMUNITY-CROSS-INSTRUCT: Unsupervised Instruction Generation for Aligning Large Language Models to Online Communities](https://arxiv.org/pdf/2406.12074)
**Summary**: The paper introduces Community-Cross-Instruct, an unsupervised framework for aligning large language models (LLMs) to online communities by automatically generating instruction-output pairs from community discussions. This method allows for the fine-tuning of foundational LLMs to accurately represent and evaluate the beliefs of specific communities, demonstrated through applications on Reddit. Unlike previous methods, it eliminates the need for human-authored instructions, making the process more scalable and applicable across various domains.

### 119. [Self-MoE: Towards Compositional Large Language Models with Self-Specialized Experts](https://arxiv.org/pdf/2406.12034)
**Summary**: The paper introduces Self-MoE, a method that converts monolithic large language models (LLMs) into modular systems of self-specialized experts (MiXSE) using self-generated synthetic data. This approach enhances the LLM's performance across various tasks without requiring extensive human-labeled data, showing significant improvements (6.5% on average) in benchmarks and outperforming other methods. The study emphasizes the benefits of modularity and self-improvement in creating efficient and adaptable LLMs.

### 120. [When Parts Are Greater Than Sums: Individual LLM Components Can Outperform Full Models](https://arxiv.org/pdf/2406.13131)
**Summary**: The paper investigates in-context learning by analyzing the individual contributions of attention heads and MLPs within large language models. It identifies various types of components—good, bad, and label-biased—and demonstrates that reweighting these components based on a few labeled examples can significantly improve model performance. The study offers insights into the internal workings of language models and presents a practical method for enhancing in-context learning accuracy.

### 121. [FastMem: Fast Memorization of Prompt Improves Context Awareness of Large Language Models](https://arxiv.org/pdf/2406.16069)
**Summary**: The paper introduces FastMem, a method to improve the context awareness of large language models (LLMs) by quickly memorizing the prompt. By optimizing only the last Feed-Forward Network module, FastMem enhances the model's ability to comprehend and follow context accurately, leading to significant improvements in tasks like reading comprehension and text summarization. The method shows notable gains in accuracy and reduces output structure failures, demonstrating its potential to enhance LLM reliability.

### 122. [SoK: Membership Inference Attacks on LLMs are Rushing Nowhere (and How to Fix It)](https://arxiv.org/pdf/2406.17975)
**Summary**: The paper reviews recent Membership Inference Attacks (MIAs) on Large Language Models (LLMs) and highlights concerns about distribution shifts in post-hoc datasets, questioning the validity of current evaluations. It proposes new methods for more robust evaluation, including randomized test splits and unique sequence injections, to better assess LLM memorization and guide future MIA research.

### 123. [Detection and Measurement of Syntactic Templates in Generated Text](https://arxiv.org/pdf/2407.00211)
**Summary**: The paper introduces syntactic templates as a method to analyze repetition in text generated by large language models (LLMs), finding that models frequently produce templated text not seen in human-authored texts. It reveals that 76% of these templates originate from pre-training data and persist through fine-tuning, making them useful for evaluating model diversity and style memorization.

### 124. [Make Some Noise: Unlocking Language Model Parallel Inference Capability through Noisy Training](https://arxiv.org/pdf/2406.17404)
**Summary**: The paper introduces the Make Some Noise (MSN) training framework, which enhances language model parallel decoding by introducing noise during training, eliminating the need for additional model structures or supervised fine-tuning. The authors also propose a tree-based retrieval-augmented Jacobi (TR-Jacobi) decoding strategy to further speed up inference. Experiments demonstrate that MSN improves inference speed by 2.3-2.7x without compromising model performance, achieving comparable acceleration to state-of-the-art models with additional structures.

### 125. [Self-training Language Models for Arithmetic Reasoning](https://arxiv.org/pdf/2407.08400)
**Summary**: The paper investigates the effectiveness of self-training language models for arithmetic reasoning without additional annotated data, using automated feedback. It finds significant improvements in both offline and online self-training scenarios, with preference optimization methods outperforming traditional supervised training in online settings due to better stability and robustness.

### 126. [A Survey on Natural Language Counterfactual Generation](https://arxiv.org/pdf/2407.03993)
**Summary**: The paper surveys natural language counterfactual generation, a technique that modifies text to change its classification, offering insights into model predictions and enhancing robustness. It categorizes methods into four groups and reviews evaluation metrics, highlighting ongoing challenges and future research directions.

### 127. [DogeRM: Equipping Reward Models with Domain Knowledge through Model Merging](https://arxiv.org/pdf/2407.01470)
**Summary**: The paper introduces DogeRM, a framework that merges domain-specific knowledge into general reward models to improve performance in reinforcement learning from human feedback (RLHF). By integrating expert annotations through model merging, DogeRM reduces the need for costly preference data collection and demonstrates enhanced performance across various benchmarks, highlighting its potential for more efficient model alignment.

### 128. [To Forget or Not? Towards Practical Knowledge Unlearning for Large Language Models](https://arxiv.org/pdf/2407.01920)
**Summary**: The paper introduces KnowUnDo, a benchmark for evaluating knowledge unlearning in Large Language Models (LLMs), focusing on the risk of excessive unlearning of sensitive data. It proposes MemFlex, a method that uses gradient information to precisely target and unlearn sensitive parameters, demonstrating superior performance in retaining general knowledge while effectively unlearning sensitive information.

### 129. [OffsetBias: Leveraging Debiased Data for Tuning Evaluators](https://arxiv.org/pdf/2407.06551)
**Summary**: The paper identifies six types of biases in judge models used to evaluate generated responses from Large Language Models (LLMs) and introduces EvalBiasBench, a collection of test cases to assess these biases. The authors propose methods to construct a debiased dataset, OffsetBias, and demonstrate that fine-tuning on this dataset improves the robustness and performance of judge models in various evaluation scenarios.

### 130. [MMedAgent: Learning to Use Medical Tools with Multi-modal Agent](https://arxiv.org/pdf/2407.02483)
**Summary**: The paper introduces MMedAgent, the first multi-modal agent specifically designed for the medical field, which leverages a curated dataset of medical tools to select the most appropriate tool for various tasks. Experimental results show that MMedAgent outperforms existing open-source and closed-source models, including GPT-4, in handling diverse medical tasks, while also demonstrating efficiency in integrating new tools.

### 131. [NativQA: Multilingual Culturally-Aligned Natural Query for LLMs](https://arxiv.org/pdf/2407.09823)
**Summary**: The paper introduces NativQA, a scalable, language-independent framework for creating culturally and regionally aligned QA datasets in native languages, addressing the lack of such resources for evaluating and fine-tuning large language models (LLMs). The authors demonstrate the framework's efficacy with MultiNativQA, a multilingual dataset of 64k QA pairs in seven languages, and benchmark LLMs, highlighting the framework's utility for low-resource and dialectally-rich languages.

### 132. [Historical Ink: 19th Century Latin American Spanish Newspaper Corpus with LLM OCR Correction](https://arxiv.org/pdf/2407.12838)
**Summary**: The paper introduces a novel dataset of 19th-century Latin American newspaper texts, filling a significant gap in historical and linguistic research. Additionally, it presents a flexible framework using a Large Language Model for OCR error correction and linguistic analysis, which is applied to the new dataset, enhancing the accuracy and usability of historical texts.

### 133. [Cactus: Towards Psychological Counseling Conversations using Cognitive Behavioral Theory](https://arxiv.org/pdf/2407.03103)
**Summary**: The paper introduces Cactus, a multi-turn dialogue dataset designed to simulate real-life psychological counseling sessions using Cognitive Behavioral Therapy (CBT). By creating diverse client personas and systematically applying CBT techniques, Cactus aims to address the lack of realistic counseling datasets for training open-source large language models (LLMs). Experimental results show that a model trained with Cactus, named Camel, outperforms other models in counseling skills, indicating its potential as a counseling agent.

### 134. [Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/pdf/2407.10930)
**Summary**: The paper introduces a novel approach called BetterTogether that combines fine-tuning and prompt optimization to enhance the performance of modular NLP pipelines, particularly in scenarios where intermediate labels or gradient flow are absent. By alternating between optimizing language model weights and prompt templates, the method achieves significant improvements in downstream task metrics, outperforming individual optimizations by up to 60% and 6% on average across various models and tasks.

### 135. [Unlocking the Potential of Model Merging for Low-Resource Languages](https://arxiv.org/pdf/2407.03994)
**Summary**: The paper introduces model merging as an alternative to the conventional continual pre-training and supervised fine-tuning approach for adapting large language models (LLMs) to low-resource languages. By combining models with distinct capabilities, the method effectively enhances task-solving abilities in LLMs for low-resource languages without requiring additional training data in the target languages. Experiments with Llama-2-7B show that model merging outperforms the CT-then-SFT approach, especially in data-scarce scenarios, and the introduction of a slack variable in the merging algorithm further improves performance.

### 136. [Knowledge Mechanisms in Large Language Models: A Survey and Perspective](https://arxiv.org/pdf/2407.15017)
**Summary**: The paper surveys knowledge mechanisms in Large Language Models (LLMs), categorizing them into knowledge utilization and evolution. It explores how LLMs memorize, comprehend, apply, and create knowledge, as well as how knowledge evolves within individual and group models. The study also addresses the fragility of parametric knowledge and hypothesizes about potential "dark knowledge" challenges, aiming to guide future research on understanding and improving LLMs.

### 137. [A Comparison of Language Modeling and Translation as Multilingual Pretraining Objectives](https://arxiv.org/pdf/2407.15489)
**Summary**: The paper compares multilingual pretraining objectives, focusing on language modeling and translation, in a controlled environment to establish best practices. It finds that the choice of pretraining objective is influenced by the model architecture and that multilingual translation is highly effective under suitable conditions. The study emphasizes the importance of comparable training data and model architectures for meaningful comparisons.

### 138. [Optimal and efficient text counterfactuals using Graph Neural Networks](https://arxiv.org/pdf/2408.01969)
**Summary**: The paper introduces a framework for generating counterfactual explanations in NLP models using Graph Neural Networks, which creates semantically edited inputs that alter model predictions. The framework is tested on binary sentiment and topic classification tasks, demonstrating that it produces contrastive, fluent, and minimal edits while being significantly faster than existing methods.

### 139. [Revisiting Who's Harry Potter: Towards Targeted Unlearning from a Causal Intervention Perspective](https://arxiv.org/pdf/2407.16997)
**Summary**: The paper revisits the Who's Harry Potter (WHP) method for Large Language Model (LLM) unlearning, proposing a new task of targeted unlearning where only specific information about a given target is removed from the model. It introduces a causal intervention framework to model the unlearning process, treating the target's knowledge as a confounder and the unlearning as a deconfounding process. The proposed approach demonstrates competitive performance across various criteria without explicit optimization, with code made publicly available.

### 140. [CMR Scaling Law: Predicting Critical Mixture Ratios for Continual Pre-training of Language Models](https://arxiv.org/pdf/2407.17467)
**Summary**: The paper introduces the Critical Mixture Ratio (CMR) scaling law, which predicts the optimal balance between general and domain-specific data for continual pre-training of large language models (LLMs) to prevent catastrophic forgetting and enhance domain-specific performance. The study reveals a power-law relationship between loss, mixture ratio, and training tokens, providing a practical guideline for efficient and effective LLM training in specialized domains.

### 141. [DYNAMICQA: Tracing Internal Knowledge Conflicts in Language Models](https://arxiv.org/pdf/2407.17023)
**Summary**: The paper introduces DynamicQA, a novel dataset designed to study intra-memory conflicts in Language Models (LMs), where conflicting knowledge within the model's parameters affects its ability to integrate new context. The dataset includes facts with temporal and disputable dynamics, allowing for the evaluation of semantic entropy and a new coherent persuasion score. The study finds that LMs exhibit more intra-memory conflict with dynamic facts and that these conflicts hinder the model's ability to update with new context, suggesting challenges for retrieval-augmented generation.

### 142. [HySem: A context length optimized LLM pipeline for unstructured tabular extraction](https://arxiv.org/pdf/2408.09434)
**Summary**: The paper introduces HySem, a pipeline designed to extract and semantically represent tabular data from HTML tables, addressing the challenges of diverse table presentations and context length limitations in Large Language Models (LLMs). By optimizing context length and using a custom fine-tuned model, HySem achieves higher accuracy and performance compared to open-source models and is competitive with OpenAI GPT-4, making it suitable for cost- and privacy-sensitive pharmaceutical enterprises.

### 143. [Deconfounded Causality-aware Parameter-Efficient Fine-Tuning for Problem-Solving Improvement of LLMs](https://arxiv.org/pdf/2409.02686)
**Summary**: The paper investigates the reasoning limitations of Large Language Models (LLMs) and proposes a novel parameter-efficient fine-tuning method called Deconfounded Causal Adaptation (DCA) to enhance their problem-solving capabilities. By formulating the reasoning process into a causal framework and visualizing the text generation, the authors demonstrate that DCA significantly improves LLM performance across benchmarks with minimal tunable parameters, achieving better or comparable results to other fine-tuning methods.

### 144. [Enhancing adversarial robustness in Natural Language Inference using explanations](https://arxiv.org/pdf/2409.07423)
**Summary**: The paper investigates enhancing adversarial robustness in Natural Language Inference (NLI) by using natural language explanations as a defense strategy. By fine-tuning a classifier on explanations rather than premise-hypothesis pairs, the authors achieve improved robustness against adversarial attacks compared to traditional methods. The study also explores the correlation between language generation metrics and human perception to validate the semantic validity of generated explanations.

### 145. [IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS](https://arxiv.org/pdf/2409.05356)
**Summary**: The paper introduces IndicVoices-R (IV-R), a massive multilingual Indian TTS dataset derived from ASR data, featuring 1,704 hours of high-quality speech from 10,496 speakers across 22 languages. The authors demonstrate improved zero-shot speaker generalization by fine-tuning an English pre-trained model on a combined dataset of IndicTTS and IV-R, addressing the scarcity of high-quality TTS data for Indian languages.

### 146. [WinoPron: Revisiting English Winogender Schemas for Consistency, Coverage, and Grammatical Case](https://arxiv.org/pdf/2409.05653)
**Summary**: The paper "WinoPron: Revisiting English Winogender Schemas for Consistency, Coverage, and Grammatical Case" addresses issues in the Winogender Schemas dataset, which is used to evaluate gender bias in coreference resolution. The authors identify and correct problems such as inconsistent pronoun treatment, template violations, and typographical errors, resulting in a new dataset called WinoPron. They use WinoPron to evaluate coreference resolution models, finding that accusative pronouns are particularly challenging, and introduce a new method to assess pronominal bias beyond binary gender distinctions.

### 147. [Beyond Persuasion: Towards Conversational Recommender System with Credible Explanations](https://arxiv.org/pdf/2409.14399)
**Summary**: The paper introduces PC-CRS, a method designed to enhance the credibility of explanations in conversational recommender systems (CRS) by integrating credibility-aware persuasive strategies and post-hoc self-reflection. The approach aims to balance persuasion with trustworthiness, demonstrated through experiments showing improved credibility and potential to enhance recommendation accuracy.

### 148. [Obliviate: Neutralizing Task-agnostic Backdoors within the Parameter-efficient Fine-tuning Paradigm](https://arxiv.org/pdf/2409.14119)
**Summary**: The paper introduces Obliviate, a defense mechanism for neutralizing task-agnostic backdoors in parameter-efficient fine-tuning (PEFT) of large language models. The proposed method employs techniques to amplify benign neurons and penalize trigger tokens, effectively reducing the success rate of state-of-the-art backdoor attacks by 83.6%. Obliviate also demonstrates robust defense against task-specific backdoors and adaptive attacks.

### 149. [Visual Question Decomposition on Multimodal Large Language Models](https://arxiv.org/pdf/2409.19339)
**Summary**: The paper investigates the question decomposition capability of Multimodal Large Language Models (MLLMs) and introduces a systematic evaluation framework to assess the quality of decomposed sub-questions. It identifies limitations in existing MLLMs and proposes a finetuning dataset, DecoVQA+, and an efficient finetuning pipeline to enhance the models' ability to produce high-quality sub-questions and perform selective decomposition, leading to improved accuracy on VQA benchmarks.

### 150. [PEAR: Position-Embedding-Agnostic Attention Re-weighting Enhances Retrieval-Augmented Generation with Zero Inference Overhead](https://arxiv.org/pdf/2409.19745)
**Summary**: The paper introduces PEAR, a method that enhances the context awareness of large language models (LLMs) in retrieval-augmented generation (RAG) tasks without any inference overhead. PEAR identifies and re-weights attention heads that suppress context awareness, optimizing their impact through learnable coefficients. This approach improves RAG performance across various tasks while remaining agnostic to position embedding algorithms, making it both efficient and broadly applicable.

### 151. [Aligning with Logic: Measuring, Evaluating and Improving Logical Consistency in Large Language Models](https://arxiv.org/pdf/2410.02205)
**Summary**: The paper introduces a framework to measure and improve the logical consistency of Large Language Models (LLMs), which is crucial for their reliability and trustworthiness. It proposes three fundamental proxies—transitivity, commutativity, and negation invariance—to quantify logical consistency and demonstrates its impact on LLM robustness. Additionally, the paper presents a data refinement technique to enhance logical consistency while maintaining alignment with human preferences, showing its effect on LLM-based logic-dependent algorithms.

### 152. [Measuring and Improving Persuasiveness of Large Language Models](https://arxiv.org/pdf/2410.02653)
**Summary**: The paper introduces PersuasionBench and PersuasionArena, the first large-scale benchmarks for measuring the persuasiveness of large language models (LLMs). It finds that while larger models tend to be more persuasive, targeted training can significantly enhance the persuasive capabilities of smaller models, challenging the assumption that scale alone determines effectiveness. The study emphasizes the need for more comprehensive metrics to assess AI's societal impact, beyond computational power.

### 153. [OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data](https://arxiv.org/pdf/2410.01560)
**Summary**: The paper introduces OpenMathInstruct-2, a massive open-source dataset for mathematical reasoning, consisting of 14M question-solution pairs, significantly larger than previous datasets. Through ablation experiments, the authors identify key factors affecting dataset quality, such as solution format and question diversity, and demonstrate that finetuning with OpenMathInstruct-2 improves performance on the MATH benchmark by 15.9%. The dataset, code, and models are released under a permissive license to support further research.

### 154. [Better Instruction-Following Through Minimum Bayes Risk](https://arxiv.org/pdf/2410.02902)
**Summary**: The paper explores using Minimum Bayes Risk (MBR) decoding with reference-based LLM judges to improve the performance of instruction-following LLMs, finding significant gains over traditional decoding methods. Additionally, the authors investigate iterative self-training on MBR-decoded outputs, which leads to performance improvements that often match or exceed the base models' MBR decoding performance.

### 155. [Efficient Model-Agnostic Multi-Group Equivariant Networks](https://arxiv.org/pdf/2310.09675)
**Summary**: The paper introduces efficient model-agnostic equivariant network designs for scenarios with multiple input groups or large product groups acting on a single input. It proposes novel fusion layers called IS layers that satisfy invariance-symmetry constraints and demonstrates their universality in approximating invariant-symmetric functions. The designs are shown to be computationally efficient and competitive with existing methods in various applications, including multi-image classification, language compositionality, fairness in natural language generation, and robust zero-shot image classification.

### 156. [READ: Recurrent Adapter with Partial Video-Language Alignment for Parameter-Efficient Transfer Learning in Low-Resource Video-Language Modeling](https://arxiv.org/pdf/2312.06950)
**Summary**: The paper introduces READ, a novel Recurrent Adapter with Partial Video-Language Alignment (PVLA) for parameter-efficient transfer learning in low-resource video-language modeling. READ employs recurrent computation to capture temporal relations and uses partial optimal transport to maintain task-related information, significantly outperforming existing fine-tuning strategies on multiple benchmarks.

### 157. [Rethinking the Role of Proxy Rewards in Language Model Alignment](https://arxiv.org/pdf/2402.03469)
**Summary**: The paper investigates the role of proxy rewards in aligning Large Language Models (LLMs) with human values through a method called "reverse reward engineering." By creating interpretable features as a white-box reward function, the authors aim to replicate the ground truth reward signal, finding that successful emulation requires responses to be relevant, sufficiently long for open-ended questions, and consistent for closed-ended questions. The resulting models demonstrate competitive performance in alignment benchmarks, suggesting the potential of this approach as a strong baseline for LLM alignment without the need for explicit human feedback or reward model training.

### 158. [Self-Play Preference Optimization for Language Model Alignment](https://arxiv.org/pdf/2405.00675)
**Summary**: The paper introduces Self-Play Preference Optimization (SPPO), a novel method for aligning language models by treating the alignment problem as a constant-sum two-player game. SPPO iteratively updates policies to approximate the Nash equilibrium, achieving state-of-the-art performance in preference-based evaluations without relying on additional external supervision. The method demonstrates superior results on multiple benchmarks, outperforming existing approaches like DPO and IPO.

### 159. [Efficient Prompting for LLM-based Generative Internet of Things](https://arxiv.org/pdf/2406.10382)
**Summary**: The paper introduces a LLM-based Generative IoT (GIoT) system designed for local network deployment, addressing the limitations of open-source LLMs by employing prompt engineering and a modular approach to manage and process prompts. The system is demonstrated through a Table Question Answering task, showing competitive performance compared to state-of-the-art LLMs, and is adaptable to new tasks without additional training.

### 160. [mDPO: Conditional Preference Optimization for Multimodal Large Language Models](https://arxiv.org/pdf/2406.11839)
**Summary**: The paper introduces mDPO, a novel approach to conditional preference optimization for multimodal large language models (LLMs), addressing the issue of unconditional preference in multimodal scenarios. By incorporating image preferences and introducing a reward anchor, mDPO significantly enhances model performance, particularly in reducing hallucination, as demonstrated across various benchmarks.

### 161. [WellDunn: On the Robustness and Explainability of Language Models and Large Language Models in Identifying Wellness Dimensions](https://arxiv.org/pdf/2406.12058)
**Summary**: The paper evaluates the robustness and explainability of Language Models (LMs) and Large Language Models (LLMs) in identifying Wellness Dimensions, focusing on attention fidelity and alignment with expert-labeled explanations. It finds that despite human-like capabilities, GPT-3.5/4 and MedAlpaca underperform in both performance and explanation quality, with low alignment between attention mechanisms and ground truth explanations, emphasizing the need for improved consistency and domain-specific knowledge in mental health applications.

### 162. [Unlocking Continual Learning Abilities in Language Models](https://arxiv.org/pdf/2406.17245)
**Summary**: The paper introduces MIGU, a rehearsal-free and task-label-free method for continual learning in language models, which updates model parameters based on the magnitude of outputs in linear layers, addressing the issue of catastrophic forgetting. MIGU demonstrates state-of-the-art performance across various language model architectures and continual learning benchmarks, significantly improving accuracy in multi-task scenarios.

### 163. [Can Large Language Models Understand Symbolic Graphics Programs?](https://arxiv.org/pdf/2408.08313)
**Summary**: The paper investigates the ability of large language models (LLMs) to understand symbolic graphics programs, which require spatial-semantic reasoning without relying on vision encoders. By creating a benchmark for semantic visual understanding, the authors evaluate LLMs and introduce Symbolic Instruction Tuning (SIT) to enhance their performance, finding that SIT improves both understanding of symbolic programs and general reasoning abilities.

### 164. [An Adversarial Perspective on Machine Unlearning for AI Safety](https://arxiv.org/pdf/2409.18025)
**Summary**: The paper examines the effectiveness of machine unlearning methods in AI safety, particularly against adversarial attacks. It demonstrates that existing jailbreak techniques can bypass unlearning protections when applied strategically and introduces adaptive methods that recover unlearned capabilities, questioning the robustness of current unlearning approaches compared to traditional safety training.

### 165. [CBF-LLM: Safe Control for LLM Alignment](https://arxiv.org/pdf/2408.15625)
**Summary**: The paper introduces CBF-LLM, a control-based framework that uses control barrier functions (CBFs) to align large language models (LLMs) and ensure safe text generation. By applying a safety filter to the token sequence output of a baseline LLM, the framework reduces the need for interventions while maintaining user-specified alignment. The approach is demonstrated using Llama 3 and RoBERTa models, with the source code available for further exploration.

### 166. [Representation Tuning](https://arxiv.org/pdf/2409.06927)
**Summary**: The paper introduces "representation tuning," a method for embedding behavioral vectors directly into large language models (LLMs) to control their output characteristics, such as honesty. By fine-tuning the model with a dual loss function combining cosine similarity and token-based loss, the authors demonstrate that this approach outperforms online steering and standard fine-tuning in producing more consistent and generalized honest responses.

### 167. [The Crucial Role of Samplers in Online Direct Preference Optimization](https://arxiv.org/pdf/2409.19605)
**Summary**: The paper investigates the impact of different sampling strategies on the convergence rates of Direct Preference Optimization (DPO), finding that uniform sampling leads to linear convergence, while an online sampler achieves quadratic convergence. The proposed method, incorporating posterior distributions and logit mixing, significantly outperforms existing approaches in empirical evaluations, suggesting new directions for algorithm design in language model alignment.

### 168. [Few-shot Prompting for Pairwise Ranking: An Effective Non-Parametric Retrieval Model](https://arxiv.org/pdf/2409.17745)
**Summary**: The paper introduces a pairwise few-shot ranker that enhances retrieval performance by leveraging a small number of training examples, improving upon zero-shot baselines. This non-parametric approach shows consistent improvements in both in-domain and out-domain benchmarks, achieving results close to supervised models without the need for complex training pipelines.

### 169. [Adversarial Suffixes May Be Features Too!](https://arxiv.org/pdf/2410.00451)
**Summary**: The paper investigates the hypothesis that adversarial suffixes in large language models (LLMs) are not just vulnerabilities but may represent features that can dominate model behavior. Through experiments, the authors demonstrate that benign features can be transformed into adversarial suffixes that compromise safety alignment and that these features can be introduced through fine-tuning with benign datasets alone. This suggests a critical risk from benign features in training data and emphasizes the need for further research to enhance LLM safety alignment.

### 170. [Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models](https://arxiv.org/pdf/2410.02298)
**Summary**: The paper introduces Jailbreak Antidote, a method for dynamically adjusting the safety-utility balance in large language models (LLMs) by manipulating a sparse subset of the model's internal states during inference. This approach allows for real-time safety control without increasing computational overhead or inference latency, and it is shown to be effective across a range of LLMs and against various jailbreak attacks.

### 171. [Training Nonlinear Transformers for Chain-of-Thought Inference: A Theoretical Generalization Analysis](https://arxiv.org/pdf/2410.02167)
**Summary**: The paper presents a theoretical analysis of training nonlinear Transformers for Chain-of-Thought (CoT) inference, addressing the challenges of nonconvex optimization in nonlinear attention models. It quantifies the necessary training samples and iterations for achieving CoT generalization, proving its effectiveness on unseen tasks with distribution-shifted data, and characterizing conditions for accurate reasoning even with noisy examples. This contrasts with in-context learning, which may fail where CoT succeeds.



---

*Last updated on 2024-10-09*