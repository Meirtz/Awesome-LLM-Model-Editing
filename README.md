# Awesome-LLM-Model-Editing

## Awesome LLM Model Editing

Welcome to the **Awesome LLM Model Editing** repository! This project curates a list of high-quality resources related to LLM Model Editing, including research papers, tools, libraries, and more.

## Daily Updates

### 2024-10-11

### 1. [MKGL: Mastery of a Three-Word Language](https://arxiv.org/pdf/2410.07526)

**Summary**: The paper introduces MKGL, a specialized Knowledge Graph Language (KGL) designed to integrate LLMs with knowledge graphs (KGs). By structuring sentences as entity-relation-entity triplets, MKGL minimizes hallucinations and enhances accuracy. The study demonstrates that LLMs can achieve fluency in KGL through tailored learning methods, significantly outperforming traditional KG embedding techniques in KG completion tasks.

### 2. [KRAG Framework for Enhancing LLMs in the Legal Domain](https://arxiv.org/pdf/2410.07551)

**Summary**: The paper introduces the Knowledge Representation Augmented Generation (KRAG) framework, which enhances Large Language Models (LLMs) by incorporating critical knowledge entities and relationships often missing in standard datasets. Specifically, in the legal domain, KRAG is implemented through Soft PROLEG, which uses inference graphs to improve LLMs' ability to provide structured legal reasoning and explanations. This approach significantly advances natural language understanding in specialized domains like law.

### 3. [Uncovering Overfitting in Large Language Model Editing](https://arxiv.org/pdf/2410.07819)

**Summary**: The paper identifies a phenomenon called Editing Overfit in Large Language Models, where edited models overly prioritize the edit target, leading to poor generalization in complex tasks. To address this, the authors introduce a new benchmark, EVOKE, and propose a strategy called Learn to Inference (LTI) with a Multi-stage Inference Constraint module, which helps edited models recall knowledge more effectively, reducing overfitting.

### 4. [Fine-Tuning Language Models for Ethical Ambiguity: A Comparative Study of Alignment with Human Responses](https://arxiv.org/pdf/2410.07826)

**Summary**: The study investigates the alignment of language models with human judgments in morally ambiguous scenarios by fine-tuning models on curated datasets from the Scruples project. Significant improvements in model performance were observed post-fine-tuning, particularly in cross-entropy and Dirichlet scores, highlighting the potential for enhancing ethical reasoning in language models. However, the fine-tuned models still lagged behind BERT and RoBERTa in cross-entropy scores, suggesting ongoing challenges in capturing human judgment nuances.

### 5. [Disease Entity Recognition and Normalization is Improved with Large Language Model Derived Synthetic Normalized Mentions](https://arxiv.org/pdf/2410.07951)

**Summary**: The study demonstrates that fine-tuning a Large Language Model (LLM) to generate synthetic normalized mentions of disease entities significantly improves Disease Entity Normalization (DEN) performance, particularly for out-of-distribution data, while yielding only modest enhancements for Disease Entity Recognition (DER). The synthetic data augmentation led to substantial accuracy improvements in DEN across multiple datasets, suggesting that LLM-generated data can effectively supplement traditional training methods.

### 6. [A Closer Look at Machine Unlearning for Large Language Models](https://arxiv.org/pdf/2410.08109)

**Summary**: The paper examines the challenges of machine unlearning in LLMs, particularly in removing specific content while maintaining model performance. It introduces new evaluation metrics and proposes methods like maximizing entropy for untargeted unlearning and answer preservation loss for targeted unlearning to address these challenges. Experimental results show the effectiveness of these approaches across various unlearning scenarios.

### 7. [Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study over Open-ended Question Answering](https://arxiv.org/pdf/2410.08085)

**Summary**: The paper introduces OKGQA, a new benchmark for evaluating the effectiveness of Knowledge Graphs (KGs) in enhancing Large Language Models (LLMs) in open-ended question answering scenarios. The benchmark aims to assess both the reduction in hallucinations and the improvement in reasoning capabilities of LLMs when integrated with KGs, and it also includes a perturbed version (OKGQA-P) to test model robustness against KG errors. The study seeks to determine whether KGs can make LLMs more trustworthy in real-world applications and to guide future research in this area.

### 8. [Insight Over Sight? Exploring the Vision-Knowledge Conflicts in Multimodal LLMs](https://arxiv.org/pdf/2410.08145)

**Summary**: The paper investigates the issue of vision-knowledge conflicts in Multimodal Large Language Models (MLLMs), where visual information contradicts the model's internal commonsense knowledge. It introduces a benchmark with 374 images and 1,122 QA pairs to assess conflict resolution, finding that models often over-rely on textual queries. A new prompting strategy, "Focus-on-Vision" (FoV), is proposed to improve models' reliance on visual data, enhancing their conflict-resolution capabilities.

### 9. [From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions](https://arxiv.org/pdf/2410.08197)

**Summary**: The paper introduces DRAFT, a framework that dynamically refines tool documentation for Large Language Models (LLMs) by analyzing feedback and interaction trails. Through a trial-and-error approach involving experience gathering, learning, and documentation rewriting, DRAFT iteratively improves documentation quality, enhancing LLMs' understanding and use of tools. The framework's effectiveness is demonstrated through experiments showing improved documentation quality and cross-model generalization.

### 10. [Mitigating Gender Bias in Code Large Language Models via Model Editing](https://arxiv.org/pdf/2410.07820)

**Summary**: The paper introduces CodeGenBias, a dataset and FB-Score metric to evaluate gender bias in code Large Language Models (LLMs), and proposes MG-Editing, a multi-granularity model editing approach to mitigate this bias. Experiments show that MG-Editing effectively reduces gender bias while preserving code generation capabilities, with the best performance at row and neuron levels of granularity.

### 11. [Composite Learning Units: Generalized Learning Beyond Parameter Updates to Transform LLMs into Adaptive Reasoners](https://arxiv.org/pdf/2410.08037)

**Summary**: The paper introduces Composite Learning Units (CLUs) to enhance Large Language Models (LLMs) with continuous, generalized learning capabilities without traditional parameter updates. CLUs utilize a dynamic knowledge repository, including a General Knowledge Space and a Prompt-Specific Knowledge Space, to iteratively refine reasoning through goal-driven interactions and feedback, enabling adaptive reasoning and autonomous learning from past experiences.

### 12. [SLIM: Let LLM Learn More and Forget Less with Soft LoRA and Identity Mixture](https://arxiv.org/pdf/2410.07739)

**Summary**: The paper introduces SLIM, a novel mixture of expert framework that combines Soft LoRA and Identity Mixture to efficiently fine-tune LLMs while mitigating catastrophic forgetting. SLIM dynamically routes between LoRA adapters and skipping connections, enhancing out-of-domain distinction and preserving the base model's general capabilities. Experimental results show that SLIM outperforms existing parameter-efficient fine-tuning methods in balancing downstream task performance and preventing forgetting.

### 13. [Fuse to Forget: Bias Reduction and Selective Memorization through Model Fusion](https://arxiv.org/pdf/2311.07682)

**Summary**: The paper explores the use of model fusion to reduce unwanted knowledge, such as shortcuts, social biases, and memorization of training data, in fine-tuned language models. Through experiments, it shows that model fusion enhances shared knowledge while forgetting unshared knowledge, making it a promising tool for debiasing and addressing privacy concerns in language models.

### 14. [AKEW: Assessing Knowledge Editing in the Wild](https://arxiv.org/pdf/2402.18909)

**Summary**: The paper introduces AKEW, a benchmark for assessing knowledge editing in practical scenarios, addressing the limitations of current evaluations that focus on structured facts from curated datasets. AKEW includes settings for structured facts, unstructured texts, and extracted triplets, and features both counterfactual and real-world knowledge updates, revealing significant gaps between existing methods and practical needs.

### 15. [AutoRD: An Automatic and End-to-End System for Rare Disease Knowledge Graph Construction Based on Ontologies-enhanced Large Language Models](https://arxiv.org/pdf/2403.00953)

**Summary**: The paper introduces AutoRD, an end-to-end system designed to automate the construction of a knowledge graph for rare diseases by extracting entities and their relations from medical texts. Leveraging ontologies-enhanced Large Language Models, AutoRD integrates up-to-date structured knowledge and outperforms traditional methods and common LLMs in rare disease extraction tasks.

### 16. [LLaMP: Large Language Model Made Powerful for High-fidelity Materials Knowledge Retrieval and Distillation](https://arxiv.org/pdf/2401.17244)

**Summary**: The paper introduces LLaMP, a multimodal retrieval-augmented generation framework that enhances Large Language Models (LLMs) for high-fidelity materials knowledge retrieval and distillation. By integrating hierarchical reasoning-and-acting agents with computational and experimental data, LLaMP effectively reduces hallucination and biases in LLMs, demonstrating strong tool usage and self-consistency in handling complex materials science tasks. The framework also showcases capabilities in editing crystal structures and running molecular dynamics simulations, offering a nearly hallucination-free approach to materials informatics.

### 17. [Unlocking the Power of Large Language Models for Entity Alignment](https://arxiv.org/pdf/2402.15048)

**Summary**: The paper introduces ChatEA, a novel framework that leverages LLMs to enhance Entity Alignment (EA) in knowledge graphs. By translating KG structures into a format understandable by LLMs and employing a two-stage EA strategy, ChatEA improves EA accuracy by utilizing LLMs' extensive background knowledge and multi-step reasoning capabilities. Experimental results demonstrate ChatEA's superior performance over traditional methods.

### 18. [Verification and Refinement of Natural Language Explanations through LLM-Symbolic Theorem Proving](https://arxiv.org/pdf/2405.01379)

**Summary**: The paper introduces Explanation-Refiner, a neuro-symbolic framework that integrates Large Language Models (LLMs) with Theorem Provers (TPs) to verify and refine natural language explanations for Natural Language Inference (NLI). By leveraging TPs for formal validation and feedback, the framework aims to enhance the quality and logical correctness of explanations across various domains, thereby addressing the limitations of current crowd-sourced validation methods.

### 19. [ClaimBrush: A Novel Framework for Automated Patent Claim Refinement Based on Large Language Models](https://arxiv.org/pdf/2410.05575)

**Summary**: The paper introduces ClaimBrush, a framework for automated patent claim refinement using large language models. It includes a dataset of actual patent claim rewriting cases and a fine-tuned rewriting model, enhanced by preference optimization based on patent examiners' feedback. Experimental results demonstrate superior performance over heuristic baselines and zero-shot learning methods.

### 20. [Dr-LLaVA: Visual Instruction Tuning with Symbolic Clinical Grounding](https://arxiv.org/pdf/2405.19567)

**Summary**: The paper introduces Dr-LLaVA, a Vision-Language Model (VLM) designed for medical applications, which uses a novel alignment algorithm to ground its outputs in clinical reasoning. This algorithm leverages symbolic representations of medical knowledge to generate training data and create an automatic reward function, eliminating the need for human involvement and reducing costs. Dr-LLaVA demonstrates strong performance in multi-turn medical conversations, particularly in analyzing bone marrow pathology slides.

### 21. [Can Large Language Models Understand DL-Lite Ontologies? An Empirical Study](https://arxiv.org/pdf/2406.17532)

**Summary**: The paper investigates whether LLMs can understand DL-Lite ontologies, focusing on six representative tasks from syntactic and semantic perspectives. The study reveals that LLMs effectively grasp formal syntax and model-theoretic semantics but face challenges with TBox NI transitivity and large ABoxes. The findings aim to inform future knowledge engineering solutions.

### 22. [Revisiting the Superficial Alignment Hypothesis](https://arxiv.org/pdf/2410.03717)

**Summary**: The paper challenges the Superficial Alignment Hypothesis by demonstrating that post-training with increasing fine-tuning examples significantly improves language model performance across various tasks, including mathematical reasoning and multihop reasoning. The study reveals that model performance scales as a power law with the number of fine-tuning examples, indicating that post-training is crucial for enhancing reasoning abilities and integrating new knowledge, contrary to the hypothesis's claims.

### 23. [Neurosymbolic AI approach to Attribution in Large Language Models](https://arxiv.org/pdf/2410.03726)

**Summary**: The paper proposes a Neurosymbolic AI (NesyAI) approach to improve attribution in LLMs, addressing issues like hallucinations, biases, and unreliable sources by combining neural networks with structured symbolic reasoning. This integration aims to provide more reliable, interpretable, and adaptable systems for ensuring the factual accuracy and reliability of LLM outputs.

### 24. [Reasoning Elicitation in Language Models via Counterfactual Feedback](https://arxiv.org/pdf/2410.03767)

**Summary**: The paper introduces new metrics to evaluate the reasoning capabilities of language models, particularly in counterfactual scenarios, and proposes fine-tuning methods to enhance these abilities. The study evaluates the fine-tuned models across various reasoning tasks, demonstrating improved generalization and performance compared to baseline models.

### 25. [Precision Knowledge Editing: Enhancing Safety in Large Language Models](https://arxiv.org/pdf/2410.03772)

**Summary**: The paper introduces Precision Knowledge Editing (PKE), a technique that enhances the safety of LLMs by more effectively identifying and modifying toxic parameter regions. PKE, which builds on existing knowledge editing methods, uses neuron weight tracking and activation pathway tracing to achieve finer granularity in managing toxic content. Experiments show that PKE significantly reduces the attack success rate across various models while maintaining overall performance, outperforming closed-source models in terms of safety.

### 26. [Neuron-Level Sequential Editing for Large Language Models](https://arxiv.org/pdf/2410.04045)

**Summary**: The paper introduces Neuron-Level Sequential Editing (NSE), a novel method for continuously updating LLMs through multi-round editing without requiring costly retraining. NSE optimizes hidden states using original weights to prevent model failure and iteratively selects neurons for editing to mitigate forgetting, outperforming existing methods in sequential model editing tasks.

### 27. [DiDOTS: Knowledge Distillation from Large-Language-Models for Dementia Obfuscation in Transcribed Speech](https://arxiv.org/pdf/2410.04188)

**Summary**: The paper introduces DiDOTS, a method that leverages Large-Language-Models (LLMs) to obfuscate dementia indicators in speech transcripts, addressing privacy concerns without relying on large labeled datasets. DiDOTS uses knowledge distillation to create a more efficient model with significantly fewer parameters, outperforming existing methods in both privacy and utility preservation.

### 28. [Persona Knowledge-Aligned Prompt Tuning Method for Online Debate](https://arxiv.org/pdf/2410.04239)

**Summary**: The paper introduces a novel framework that leverages ChatGPT's capabilities to simulate audience personas and enhance argument quality assessment by aligning persona knowledge with smaller language models through prompt tuning. This approach significantly improves performance in debate scenarios by integrating audience-specific characteristics, marking a first in combining argument persuasiveness with audience personae.

### 29. [Toxic Subword Pruning for Dialogue Response Generation on Large Language Models](https://arxiv.org/pdf/2410.04155)

**Summary**: The paper introduces ToxPrune, a novel algorithm that prunes toxic subwords from Byte Pair Encoding (BPE) in trained Large Language Models (LLMs) to prevent the generation of toxic content. Contrary to previous findings that pruning BPE tokens can harm machine translation tasks, the authors demonstrate that ToxPrune effectively reduces toxicity in dialogue response generation and even improves the diversity of responses in models like NSFW-3B and Llama-3.1-6B.

### 30. [Entity Insertion in Multilingual Linked Corpora: The Case of Wikipedia](https://arxiv.org/pdf/2410.04254)

**Summary**: The paper introduces the task of entity insertion in information networks, focusing on Wikipedia, and addresses the challenge of locating suitable positions for new links in text without predefined anchors. The authors develop LocEI and its multilingual variant XLocEI, which outperform baseline models, including GPT-4, and demonstrate effective zero-shot performance across languages. This work is crucial for enhancing link insertion across Wikipedia's multilingual editions.

### 31. [Mechanistic Behavior Editing of Language Models](https://arxiv.org/pdf/2410.04277)

**Summary**: The paper introduces TaRot, a method for task adaptation in LLMs that uses learnable rotation matrices optimized via Bayesian Optimization to improve performance on classification and generation tasks. TaRot enhances both zero-shot and few-shot performance, with average improvements of 23.81% and 11.15% respectively across various models and tasks.

### 32. [Collapsed Language Models Promote Fairness](https://arxiv.org/pdf/2410.04472)

**Summary**: The paper investigates the relationship between Neural Collapse and fairness in language models, finding that debiased models exhibit collapsed alignment between token representations and word embeddings. This insight leads to a new fine-tuning method that enhances fairness across various debiasing techniques while maintaining performance on standard language understanding tasks.

### 33. [Wrong-of-Thought: An Integrated Reasoning Framework with Multi-Perspective Verification and Wrong Information](https://arxiv.org/pdf/2410.04463)

**Summary**: The paper introduces Wrong-of-Thought (WoT), an integrated reasoning framework designed to improve the performance of Large Language Models (LLMs) by addressing two key issues: the reliance on single verification methods and the ignorance of wrong information during reasoning. WoT incorporates multi-perspective verification to refine reasoning processes and utilizes wrong information to prevent recurring errors. Experimental results show that WoT outperforms existing methods across multiple datasets and LLMs, particularly in challenging computation tasks.

### 34. [Formality is Favored: Unraveling the Learning Preferences of Large Language Models on Data with Conflicting Knowledge](https://arxiv.org/pdf/2410.04784)

**Summary**: The study investigates how LLMs handle conflicting information in training data, finding that they prefer formal texts and those with fewer spelling errors, similar to human preferences. This preference leads to faster learning and better treatment of knowledge in data with these features, especially in larger models, and can be influenced by manipulating data consistency.

### 35. [Document-level Causal Relation Extraction with Knowledge-guided Binary Question Answering](https://arxiv.org/pdf/2410.04752)

**Summary**: The paper introduces a Knowledge-guided binary Question Answering (KnowQA) method for Event-Event Causal Relation Extraction (ECRE), addressing challenges like lack of document-level modeling and causal hallucinations. The proposed method, involving Event Structure Construction and Binary Question Answering, achieves state-of-the-art performance on the MECI dataset and demonstrates high generalizability and low inconsistency, especially with complete event structures post-fine-tuning.

### 36. [Activation Scaling for Steering and Interpreting Language Models](https://arxiv.org/pdf/2410.04962)

**Summary**: The paper explores the concept of steering language models by scaling activation vectors to correct incorrect predictions, such as flipping "Rome is in France" to "Rome is in Italy." The authors propose a three-term objective for effective, faithful, and minimal interventions, achieved through gradient-based optimization. They demonstrate that activation scaling is more minimal and interpretable than steering vectors, allowing for precise identification of model components.

### 37. [ZEBRA: Zero-Shot Example-Based Retrieval Augmentation for Commonsense Question Answering](https://arxiv.org/pdf/2410.05077)

**Summary**: The paper introduces ZEBRA, a zero-shot question answering framework that enhances commonsense reasoning by combining retrieval, case-based reasoning, and introspection without requiring additional training of the language model. ZEBRA outperforms existing methods and strong language models, achieving an average accuracy improvement of up to 4.5 points across eight benchmarks.

### 38. [Deciphering the Interplay of Parametric and Non-parametric Memory in Retrieval-augmented Language Models](https://arxiv.org/pdf/2410.05162)

**Summary**: The study investigates how Retrieval-Augmented Generation (RAG) models like \textsc{Atlas} balance parametric (internal) and non-parametric (retrieved) memory during response generation. Through causal mediation analysis, it reveals that the model prioritizes retrieved context over internal knowledge when both are available, and identifies two key mechanisms: determining context relevance and computing output representations for copying relevant information.

### 39. [Metadata Matters for Time Series: Informative Forecasting with Transformers](https://arxiv.org/pdf/2410.03806)

**Summary**: The paper introduces MetaTST, a novel approach that integrates metadata into Transformer models for time series forecasting, enhancing accuracy and interpretability. By converting unstructured metadata into structured text and encoding it with large language models, MetaTST enriches the embedding process and improves forecasting performance across diverse scenarios, achieving state-of-the-art results in both short- and long-term benchmarks.

### 40. [A Pluggable Common Sense-Enhanced Framework for Knowledge Graph Completion](https://arxiv.org/pdf/2410.04488)

**Summary**: The paper introduces a pluggable framework for knowledge graph completion (KGC) that integrates both factual and common sense information to improve inference accuracy. The framework is adaptable to different KGs, automatically generating common sense from factual triples and employing techniques like common sense-guided negative sampling and a dual scoring scheme. It demonstrates superior performance and scalability across various KGC tasks compared to existing models.

### 41. [Augmenting Black-box LLMs with Medical Textbooks for Biomedical Question Answering (Published in Findings of EMNLP 2024)](https://arxiv.org/pdf/2309.02233)

**Summary**: The paper introduces LLM-AMT, a system that enhances LLMs like ChatGPT with medical textbooks to improve their performance in biomedical question answering. By integrating authoritative medical knowledge through specialized modules, LLM-AMT significantly boosts response accuracy, outperforming even specialized medical models like Med-PaLM 2. The study highlights that medical textbooks are more effective than Wikipedia as a knowledge source in this domain.

### 42. [Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue](https://arxiv.org/pdf/2401.04700)

**Summary**: The paper investigates the side effects of model editing on LLMs, finding that while editing improves factuality, it often degrades general abilities like reasoning and question answering. To address this, the authors propose RECT, a regularization method that constrains the complexity of weight updates, effectively mitigating side effects while preserving editing performance.

### 43. [Corrective Retrieval Augmented Generation](https://arxiv.org/pdf/2401.15884)

**Summary**: The paper introduces Corrective Retrieval Augmented Generation (CRAG), a method designed to enhance the robustness of retrieval-augmented generation (RAG) by incorporating a retrieval evaluator to assess the quality of retrieved documents and trigger appropriate actions. CRAG also employs large-scale web searches to augment retrieval results and uses a decompose-then-recompose algorithm to focus on key information, improving performance across various RAG-based approaches in both short- and long-form generation tasks.

### 44. [Editing Conceptual Knowledge for Large Language Models](https://arxiv.org/pdf/2403.06259)

**Summary**: The paper introduces a novel approach to editing conceptual knowledge in Large Language Models (LLMs) by creating the ConceptEdit benchmark dataset and new evaluation metrics. It finds that while existing editing methods can modify concept-level definitions, they often distort related instantial knowledge, highlighting the need for improved techniques to balance these changes.

### 45. [FAC$^2$E: Better Understanding Large Language Model Capabilities by Dissociating Language and Cognition](https://arxiv.org/pdf/2403.00126)

**Summary**: The paper introduces FAC$^2$E, a framework for evaluating LLMs by distinguishing between language and cognitive capabilities. It breaks down the evaluation process into three sub-steps: knowledge recall, knowledge utilization, and problem-solving, providing a detailed diagnosis of LLMs' performance. The study identifies a common weakness in knowledge utilization and suggests a knowledge-enhanced method to improve LLM performance.

### 46. [Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems](https://arxiv.org/pdf/2402.17840)

**Summary**: The paper investigates the vulnerability of Retrieval-Augmented Generation (RAG) systems, particularly those using instruction-tuned Language Models (LMs), to datastore leakage through prompt injection. The study reveals that adversaries can exploit these systems to extract verbatim text data from the datastore, with the risk increasing as model size scales up. The authors demonstrate successful extraction from various LMs and propose mitigation strategies, including position bias elimination, to reduce this vulnerability.

### 47. [Red Teaming Language Models for Processing Contradictory Dialogues](https://arxiv.org/pdf/2405.10128)

**Summary**: The paper introduces a novel task for processing contradictory dialogues in language models, aiming to detect and modify self-contradictory statements. A Red Teaming framework is developed using a dataset of contradictory dialogues with explanatory labels, which improves detection, explanation, and modification of contradictions. The study underscores the significance of addressing logical inconsistencies in conversational AI.

### 48. [Student Data Paradox and Curious Case of Single Student-Tutor Model: Regressive Side Effects of Training LLMs for Personalized Learning](https://arxiv.org/pdf/2404.15156)

**Summary**: The paper identifies the "Student Data Paradox," where training Large Language Models (LLMs) on extensive student-tutor dialogue datasets to personalize education leads to a decline in the models' factual knowledge and reasoning abilities. The study demonstrates this paradox through quantitative analysis and introduces "hallucination tokens" as a partial solution, highlighting the ongoing challenge of balancing accurate student behavior modeling with maintaining the LLM's educational integrity.

### 49. [Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete Knowledge Graph Question Answering](https://arxiv.org/pdf/2404.14741)

**Summary**: The paper introduces Generate-on-Graph (GoG), a training-free method that leverages Large Language Models (LLMs) to address Incomplete Knowledge Graph Question Answering (IKGQA) by generating new factual triples when the provided Knowledge Graph (KG) is insufficient. GoG operates through a Thinking-Searching-Generating framework, treating LLMs as both agents and KGs, and outperforms previous methods in experimental evaluations on two datasets.

### 50. [Exploring the Compositional Deficiency of Large Language Models in Mathematical Reasoning](https://arxiv.org/pdf/2405.06680)

**Summary**: The paper investigates the compositionality of LLMs in mathematical reasoning by introducing logical traps into problem descriptions, creating a dataset called MathTrap. The study finds that while LLMs possess the necessary mathematical knowledge, they fail to spontaneously combine it with knowledge of logical traps to solve novel cases. Performance can be passively improved through external interventions like natural language prompts and fine-tuning, but systematic compositionality remains a challenge for LLMs.

### 51. [Representation noising effectively prevents harmful fine-tuning on LLMs](https://arxiv.org/pdf/2405.14577)

**Summary**: The paper introduces Representation Noising (RepNoise), a defense mechanism against harmful fine-tuning attacks on LLMs. RepNoise removes information about harmful representations, making it difficult for attackers to recover them during fine-tuning, even when they have access to the model weights. The method is shown to generalize across different types of harm and does not impair the model's performance on harmless tasks.

### 52. [WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models](https://arxiv.org/pdf/2405.14768)

**Summary**: The paper introduces WISE, a novel approach for lifelong model editing in LLMs that addresses the challenge of balancing reliability, generalization, and locality when updating knowledge. WISE employs a dual parametric memory scheme, with a main memory for pretrained knowledge and a side memory for edited knowledge, along with a router to manage queries. A knowledge-sharding mechanism ensures that edits are stored in distinct subspaces, preventing conflicts and enabling effective merging into a shared memory. Experiments demonstrate WISE's superior performance in various LLM architectures, including GPT, LLaMA, and Mistral, across different settings.

### 53. [RLSF: Reinforcement Learning via Symbolic Feedback](https://arxiv.org/pdf/2405.16661)

**Summary**: The paper introduces Reinforcement Learning via Symbolic Feedback (RLSF), a novel fine-tuning method for Large Language Models (LLMs) that leverages reasoning tools to provide detailed, token-level feedback through poly-sized certificates, addressing limitations of traditional RLHF methods. RLSF-based fine-tuning demonstrates superior performance across various tasks, enabling smaller LLMs to outperform larger, closed-source models like GPT-4.

### 54. [Self-MoE: Towards Compositional Large Language Models with Self-Specialized Experts](https://arxiv.org/pdf/2406.12034)

**Summary**: The paper introduces Self-MoE, a method that converts monolithic LLMs into modular systems of self-specialized experts (MiXSE) using self-generated synthetic data. This approach enhances the LLM's performance across various tasks without requiring extensive human-labeled data, showing significant improvements (6.5% on average) in benchmarks and outperforming other methods in flexibility and interpretability. The study underscores the importance of modularity and the potential for self-improvement in creating efficient and adaptable systems.

### 55. [DogeRM: Equipping Reward Models with Domain Knowledge through Model Merging](https://arxiv.org/pdf/2407.01470)

**Summary**: The paper introduces DogeRM, a framework that merges domain-specific knowledge into general reward models to improve performance in reinforcement learning from human feedback (RLHF). By integrating expert annotations through model merging, DogeRM reduces the need for costly preference data collection and enhances model alignment across various benchmarks.

### 56. [To Forget or Not? Towards Practical Knowledge Unlearning for Large Language Models](https://arxiv.org/pdf/2407.01920)

**Summary**: The paper introduces KnowUnDo, a benchmark for evaluating knowledge unlearning in Large Language Models (LLMs), focusing on the risk of excessive unlearning of sensitive data. It proposes MemFlex, a method that uses gradient information to precisely target and unlearn sensitive parameters, demonstrating superior performance in retaining general knowledge while effectively unlearning sensitive information.

### 57. [FastMem: Fast Memorization of Prompt Improves Context Awareness of Large Language Models](https://arxiv.org/pdf/2406.16069)

**Summary**: The paper introduces FastMem, a method to improve the context awareness of LLMs by quickly memorizing the prompt. By optimizing only the last Feed-Forward Network module, FastMem enhances the model's ability to comprehend and follow context, leading to significant improvements in tasks like reading comprehension and text summarization. The method shows notable accuracy gains and reduced output structure failures in experiments with various LLMs.

### 58. [A Survey on Natural Language Counterfactual Generation](https://arxiv.org/pdf/2407.03993)

**Summary**: The paper surveys natural language counterfactual generation, a technique that modifies texts to change their classification outcomes, offering insights into model predictions and enhancing robustness. It categorizes methods into four groups and reviews evaluation metrics, highlighting ongoing challenges and future research directions.

### 59. [MMedAgent: Learning to Use Medical Tools with Multi-modal Agent](https://arxiv.org/pdf/2407.02483)

**Summary**: The paper introduces MMedAgent, the first multi-modal agent specifically designed for the medical field, which leverages a curated dataset of medical tools to select the most appropriate tool for various tasks. Experimental results show that MMedAgent outperforms state-of-the-art open-source methods and even GPT-4o in medical task performance, while also demonstrating efficiency in integrating new tools.

### 60. [Knowledge-based Consistency Testing of Large Language Models](https://arxiv.org/pdf/2407.12830)

**Summary**: The paper introduces KonTest, an automated framework for evaluating the consistency and knowledge gaps of Large Language Models (LLMs) using a knowledge graph. KonTest identifies inconsistencies and knowledge gaps in LLMs, achieving a 19.2% error rate and revealing a 16.5% knowledge gap across tested models. The framework's mitigation method reduces knowledge gaps by 32.48%, and an ablation study highlights GPT3.5's limited effectiveness in knowledge construction.

### 61. [Knowledge Mechanisms in Large Language Models: A Survey and Perspective](https://arxiv.org/pdf/2407.15017)

**Summary**: The paper surveys knowledge mechanisms in Large Language Models (LLMs), categorizing them into knowledge utilization and evolution. It explores how LLMs memorize, comprehend, apply, and create knowledge, as well as how knowledge evolves within individual and group models. The study also addresses the fragility of parametric knowledge and hypothesizes about potential "dark knowledge" challenges, aiming to guide future research on understanding and improving LLMs.

### 62. [Revisiting Who's Harry Potter: Towards Targeted Unlearning from a Causal Intervention Perspective](https://arxiv.org/pdf/2407.16997)

**Summary**: The paper revisits the Who's Harry Potter (WHP) method for Large Language Model (LLM) unlearning, proposing a new task of targeted unlearning where only specific information about a given target is removed from the model. It introduces a causal intervention framework to model the unlearning process, treating the target's knowledge as a confounder and the unlearning as a deconfounding process. The proposed approach demonstrates competitive performance in experiments without explicit optimization for specific unlearning criteria.

### 63. [Optimal and efficient text counterfactuals using Graph Neural Networks](https://arxiv.org/pdf/2408.01969)

**Summary**: The paper introduces a framework using Graph Neural Networks to generate optimal and efficient text counterfactuals, which are semantically edited inputs that alter model predictions, thereby enhancing interpretability. The framework is tested on binary sentiment and topic classification tasks, demonstrating that it produces contrastive, fluent, and minimal edits while significantly outperforming existing methods in terms of speed.

### 64. [DYNAMICQA: Tracing Internal Knowledge Conflicts in Language Models](https://arxiv.org/pdf/2407.17023)

**Summary**: The paper introduces DynamicQA, a novel dataset designed to study intra-memory conflicts in Language Models (LMs), where conflicting knowledge within the model's parameters affects its ability to integrate new context. The dataset includes facts with temporal and disputable dynamics, allowing for the analysis of how LMs handle knowledge updates and conflicts. The study finds that LMs struggle more with updating dynamic facts, indicating challenges for retrieval-augmented generation in managing commonly adapted information.

### 65. [Deconfounded Causality-aware Parameter-Efficient Fine-Tuning for Problem-Solving Improvement of LLMs](https://arxiv.org/pdf/2409.02686)

**Summary**: The paper investigates the reasoning limitations of Large Language Models (LLMs) and proposes a novel parameter-efficient fine-tuning method called Deconfounded Causal Adaptation (DCA) to enhance their problem-solving capabilities. By formulating the reasoning process into a causal framework and visualizing the text generation, the authors demonstrate that DCA significantly improves LLM performance across benchmarks with minimal tunable parameters, achieving better or comparable results to other fine-tuning methods.

### 66. [Beyond Persuasion: Towards Conversational Recommender System with Credible Explanations](https://arxiv.org/pdf/2409.14399)

**Summary**: The paper introduces PC-CRS, a method designed to enhance the credibility of explanations in conversational recommender systems (CRS) by integrating credibility-aware persuasive strategies and post-hoc self-reflection. The approach aims to balance persuasion with credibility, addressing the issue of misleading information that can erode user trust. Experimental results show that PC-CRS effectively generates both persuasive and credible explanations, potentially improving recommendation accuracy.

### 67. [PEAR: Position-Embedding-Agnostic Attention Re-weighting Enhances Retrieval-Augmented Generation with Zero Inference Overhead](https://arxiv.org/pdf/2409.19745)

**Summary**: The paper introduces PEAR, a method that enhances the context awareness of LLMs in retrieval-augmented generation (RAG) tasks without any inference overhead. PEAR identifies and re-weights attention heads that suppress context awareness, optimizing their impact through learnable coefficients. This approach improves RAG performance across various tasks while maintaining efficiency and applicability regardless of position embedding methods.

### 68. [Can Large Language Models Understand Symbolic Graphics Programs?](https://arxiv.org/pdf/2408.08313)

**Summary**: The paper investigates the ability of LLMs to understand symbolic graphics programs, which require spatial-semantic reasoning without relying on vision encoders. By creating a benchmark for semantic visual understanding, the authors evaluate LLMs and introduce Symbolic Instruction Tuning (SIT) to enhance their performance, finding that SIT improves both symbolic program understanding and general reasoning capabilities.

### 69. [An Adversarial Perspective on Machine Unlearning for AI Safety](https://arxiv.org/pdf/2409.18025)

**Summary**: The paper examines the effectiveness of unlearning methods in removing hazardous capabilities from large language models, challenging the distinction between unlearning and traditional safety post-training. It demonstrates that existing jailbreak techniques can bypass unlearning protections and introduces adaptive methods to recover unlearned capabilities, questioning the robustness of current unlearning approaches.

### 70. [Representation Tuning](https://arxiv.org/pdf/2409.06927)

**Summary**: The paper introduces "representation tuning," a method for embedding behavioral vectors directly into LLMs to control their output characteristics, such as honesty. By fine-tuning the model with a dual loss function combining cosine similarity and token-based loss, the authors demonstrate that this approach outperforms online steering and standard fine-tuning in terms of effectiveness and generalization. This method shows promise as a safety measure for LLMs.

### 71. [Adversarial Suffixes May Be Features Too!](https://arxiv.org/pdf/2410.00451)

**Summary**: The paper investigates the hypothesis that adversarial suffixes in LLMs are not just vulnerabilities but may represent features that can dominate model behavior. Through experiments, the authors demonstrate that benign features can be transformed into adversarial suffixes that compromise safety alignment and that these features can be introduced through fine-tuning with benign datasets alone. The findings underscore the need for further research to enhance LLM safety alignment.



---

*Last updated on 2024-10-11*