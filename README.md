# Awesome-LLM-Model-Editing

## Awesome LLM Model Editing

Welcome to the **Awesome LLM Model Editing** repository! This project curates a list of high-quality resources related to LLM Model Editing, including research papers, tools, libraries, and more.

## Daily Updates

### 2024-10-18

### 1. [Exploring Prompt Engineering: A Systematic Review with SWOT Analysis](https://arxiv.org/pdf/2410.12843)

**Summary**: The paper presents a systematic review and SWOT analysis of prompt engineering techniques for Large Language Models, focusing on linguistic principles to enhance AI interactions. It identifies strengths, weaknesses, opportunities, and threats of various techniques, such as template-based approaches and fine-tuning, and suggests future research directions to improve human-machine communication.

### 2. [TextLap: Customizing Language Models for Text-to-Layout Planning](https://arxiv.org/pdf/2410.12844)

**Summary**: The paper introduces TextLap, a method that customizes LLMs to generate graphical layouts from text instructions. By leveraging a specialized dataset called InsLap, TextLap effectively outperforms existing models, including GPT-4, in tasks related to image generation and graphical design.

### 3. [A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions](https://arxiv.org/pdf/2410.12837)

**Summary**: The paper provides a thorough overview of Retrieval-Augmented Generation (RAG), detailing its evolution, current advancements, and future directions. It highlights how RAG integrates retrieval mechanisms with generative models to enhance accuracy in knowledge-intensive tasks, while also addressing challenges like scalability and ethical concerns. The survey aims to guide researchers and practitioners in leveraging RAG's potential in natural language processing.

### 4. [Navigating the Cultural Kaleidoscope: A Hitchhiker's Guide to Sensitivity in Large Language Models](https://arxiv.org/pdf/2410.12880)

**Summary**: The paper addresses the critical issue of cultural sensitivity in LLMs, particularly in smaller models that may lack comprehensive cultural training data. It introduces a cultural harm test dataset and a culturally aligned preference dataset to evaluate and fine-tune LLMs, ensuring they respect diverse cultural norms. The study demonstrates that incorporating culturally aligned feedback significantly improves model behavior, making LLMs more inclusive and ethically sound in their global applications.

### 5. [Merge to Learn: Efficiently Adding Skills to Language Models with Model Merging](https://arxiv.org/pdf/2410.12937)

**Summary**: The paper explores a cost-effective method for adding new skills to language models by training on new skills separately and then merging them with the existing model, using techniques like task vectors. This "parallel-train-then-merge" approach is shown to be comparable in effectiveness to retraining the model on updated data mixtures, particularly in enhancing safety features without compromising the model's ability to refuse harmful prompts.

### 6. [Self-Pluralising Culture Alignment for Large Language Models](https://arxiv.org/pdf/2410.12971)

**Summary**: The paper introduces CultureSPA, a framework designed to align LLMs with pluralistic human values across diverse cultures. By generating culture-related questions and comparing model outputs in culture-aware and culture-unaware settings, CultureSPA identifies culture-specific instances for fine-tuning, enhancing the model's alignment with various cultures. Experiments show that CultureSPA improves cultural alignment without compromising general model performance, with further enhancements possible through advanced prompt engineering.

### 7. [LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks](https://arxiv.org/pdf/2410.13025)

**Summary**: The paper introduces "LoRA Soups," a method for merging Low-Rank Adaptation (LoRA) modules to compose skills in Large Language Models (LLMs) for practical tasks. The study demonstrates that concatenating LoRAs (CAT) outperforms existing merging techniques, achieving significant improvements in tasks like solving math-word problems and creating question-answering bots. This work highlights model merging as an efficient approach for skill composition, particularly when training data is scarce.

### 8. [The Geometry of Numerical Reasoning: Language Models Compare Numeric Properties in Linear Subspaces](https://arxiv.org/pdf/2410.13194)

**Summary**: The paper explores how LLMs use numerical attributes encoded in low-dimensional subspaces of the embedding space for logical comparison tasks. By identifying these subspaces through partial least squares regression and manipulating hidden states within them, the study demonstrates that LLMs rely on linearly encoded numerical information to make comparison decisions across various attributes.

### 9. [MCQG-SRefine: Multiple Choice Question Generation and Evaluation with Iterative Self-Critique, Correction, and Comparison Feedback](https://arxiv.org/pdf/2410.13191)

**Summary**: The paper introduces MCQG-SRefine, a framework that uses LLMs to generate high-quality multiple-choice questions (MCQG) for professional exams like the USMLE. By incorporating expert-driven prompt engineering and iterative self-critique and correction, the framework improves question quality and difficulty, and it also introduces an automatic evaluation metric to replace expert assessments, making the process more efficient and reliable.

### 10. [Breaking Chains: Unraveling the Links in Multi-Hop Knowledge Unlearning](https://arxiv.org/pdf/2410.13274)

**Summary**: The paper identifies a critical vulnerability in current unlearning techniques for LLMs, where multi-hop queries can still retain indirect references to erased knowledge. To address this, the authors propose MUNCH, an uncertainty-based approach that breaks down multi-hop queries into subquestions and uses model uncertainty to enhance unlearning effectiveness. Empirical results show that MUNCH significantly improves the removal of indirect knowledge and can be integrated with existing unlearning methods.

### 11. [Enhancing Fact Retrieval in PLMs through Truthfulness](https://arxiv.org/pdf/2410.13562)

**Summary**: The paper explores enhancing fact retrieval in Pre-trained Language Models (PLMs) by using a helper model to assess the truthfulness of inputs based on PLM hidden states. The approach improves fact retrieval by up to 33%, demonstrating the potential of leveraging hidden state representations to enhance factual knowledge extraction from PLMs.

### 12. [RAG-DDR: Optimizing Retrieval-Augmented Generation Using Differentiable Data Rewards](https://arxiv.org/pdf/2410.13509)

**Summary**: The paper introduces Differentiable Data Rewards (DDR), a method for optimizing Retrieval-Augmented Generation (RAG) systems by aligning data preferences among different RAG modules. DDR uses a rollout approach to evaluate and optimize agents within the RAG system, leading to significant improvements in performance, especially for smaller-scale LLMs that rely heavily on retrieved knowledge. The method also enhances the generation module's ability to extract key information and resolve conflicts between internal and external knowledge.

### 13. [GeoCoder: Solving Geometry Problems by Generating Modular Code through Vision-Language Models](https://arxiv.org/pdf/2410.13510)

**Summary**: The paper introduces GeoCoder, a system that enhances vision-language models' ability to solve geometry problems by generating modular code using a predefined function library. This approach ensures accurate and deterministic calculations, overcoming the limitations of traditional VLMs in handling mathematical operations and formula application. The multimodal retrieval-augmented variant, RAG-GeoCoder, further improves performance by incorporating a non-parametric memory module for function retrieval, resulting in a 16% average improvement on the GeomVerse dataset.

### 14. [LLM-Human Pipeline for Cultural Context Grounding of Conversations](https://arxiv.org/pdf/2410.13727)

**Summary**: The paper introduces a "Cultural Context Schema" to enhance NLP models' understanding of cultural norms in conversations, particularly focusing on Chinese culture. By generating and refining over 110,000 social norm descriptions using LLMs and human verification, the authors create a dataset that significantly improves the performance of downstream tasks like emotion and dialogue act detection.

### 15. [Learning Representations for Reasoning: Generalizing Across Diverse Structures](https://arxiv.org/pdf/2410.13018)

**Summary**: The paper explores methods for improving reasoning models by enabling generalization across diverse knowledge and query structures. It introduces techniques for handling unseen entities and relations in knowledge graphs and proposes solutions for multi-step queries on both knowledge graphs and text. Additionally, the paper presents systems to facilitate machine learning development on structured data, addressing memory bottlenecks and scaling issues.

### 16. [Rethinking Misalignment in Vision-Language Model Adaptation from a Causal Perspective](https://arxiv.org/pdf/2410.12816)

**Summary**: The paper addresses the misalignment issues in Vision-Language models like CLIP during adaptation to specific tasks, identifying both task and data misalignment. It introduces a structural causal model to analyze these issues and proposes Causality-Guided Semantic Decoupling and Classification (CDC) to mitigate the interference of task-irrelevant knowledge, showing consistent effectiveness across various settings.

### 17. [Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization](https://arxiv.org/pdf/2410.12949)

**Summary**: The paper explores how mechanistic interpretability can enhance the precision and robustness of knowledge editing and unlearning in large language models. By focusing on components associated with the lookup-table mechanism for factual recall, the authors demonstrate more robust unlearning and editing that resist relearning and minimize side effects, outperforming baseline methods across different datasets and models.

### 18. [MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models](https://arxiv.org/pdf/2410.13085)

**Summary**: The paper introduces MMed-RAG, a versatile multimodal retrieval-augmented generation system designed to enhance the factuality of Medical Large Vision-Language Models (Med-LVLMs) by addressing issues like factual hallucination and misalignment. The proposed system includes a domain-aware retrieval mechanism, an adaptive context selection method, and a preference fine-tuning strategy, resulting in a 43.8% average improvement in factual accuracy across various medical datasets.

### 19. [Remember, Retrieve and Generate: Understanding Infinite Visual Concepts as Your Personalized Assistant](https://arxiv.org/pdf/2410.13360)

**Summary**: The paper introduces the Retrieval Augmented Personalization (RAP) framework, which enhances multimodal large language models (MLLMs) to function as personalized assistants by integrating user-specific knowledge. The framework involves three steps: storing user-related information in a key-value database (Remember), retrieving relevant data during conversations (Retrieve), and generating personalized responses using the retrieved information (Generate). This approach allows for real-time concept updates and improves the model's ability to handle diverse tasks like personalized image captioning and question answering.

### 20. [A Unified View of Delta Parameter Editing in Post-Trained Large-Scale Models](https://arxiv.org/pdf/2410.13841)

**Summary**: The paper introduces a unified framework for understanding delta parameter editing in post-trained large-scale models, using Riemann sum approximation of the loss function to categorize methods into competitive, decreased, and improved performance classes. It extends existing techniques like DARE and BitDelta, addressing their limitations and enhancing their applicability, supported by experiments on various models.

### 21. [BLT: Can Large Language Models Handle Basic Legal Text?](https://arxiv.org/pdf/2311.09693)

**Summary**: The paper reveals that current LLMs like GPT-4 and Claude struggle with basic legal text tasks, prompting the development of a benchmark to assess their zero-shot performance. Despite initial poor results, fine-tuning on the benchmark significantly improves even small models' accuracy, suggesting potential for enhancing LLMs' reliability in legal contexts.

### 22. [CoLLEGe: Concept Embedding Generation for Large Language Models](https://arxiv.org/pdf/2403.15362)

**Summary**: The paper introduces CoLLEGe, a meta-learning framework designed to enable large language models to quickly learn new concepts from a few examples by generating flexible embeddings. CoLLEGe focuses on next word prediction, making it compatible with language model pretraining, and demonstrates successful concept learning across various real-world tasks without task-specific training.

### 23. [ActiveRAG: Autonomously Knowledge Assimilation and Accommodation through Retrieval-Augmented Agents](https://arxiv.org/pdf/2402.13547)

**Summary**: The paper introduces ActiveRAG, a multi-agent framework that enhances Retrieval-Augmented Generation (RAG) by enabling Large Language Models (LLMs) to actively engage with and learn from retrieved external knowledge. ActiveRAG includes a knowledge assimilation agent for forming understanding and a thought accommodation agent for refining responses, leading to a 10% improvement in question-answering benchmarks and better handling of noisy retrievals.

### 24. [Avoiding Copyright Infringement via Large Language Model Unlearning](https://arxiv.org/pdf/2406.10952)

**Summary**: The paper introduces Stable Sequential Unlearning (SSU), a novel framework for removing copyrighted content from Large Language Models (LLMs) as new requests for content removal arise. SSU identifies and removes specific weight updates related to copyrighted material while maintaining the model's general language capabilities, demonstrating superior performance compared to existing methods.

### 25. [A Systematic Analysis of Large Language Models as Soft Reasoners: The Case of Syllogistic Inferences](https://arxiv.org/pdf/2406.11341)

**Summary**: The paper investigates how Large Language Models (LLMs) perform on syllogistic reasoning tasks, focusing on the impact of chain-of-thought reasoning, in-context learning, and supervised fine-tuning. The study reveals that while in-context learning and fine-tuning enhance accuracy, only fine-tuning effectively reduces reasoning biases without compromising consistency, aligning with cognitive science heuristics.

### 26. [MedCare: Advancing Medical LLMs through Decoupling Clinical Alignment and Knowledge Aggregation](https://arxiv.org/pdf/2406.17484)

**Summary**: The paper introduces MedCare, a novel approach to advancing medical LLMs by decoupling clinical alignment and knowledge aggregation. Through a two-stage progressive fine-tuning pipeline, MedCare effectively integrates diverse medical knowledge while optimizing for alignment tasks, achieving state-of-the-art performance across multiple medical tasks and demonstrating significant improvements over existing models.

### 27. [Can Large Language Models Generate High-quality Patent Claims?](https://arxiv.org/pdf/2406.19465)

**Summary**: The paper investigates the performance of LLMs in generating patent claims, finding that while GPT-4 excels in producing high-quality first independent claims, its performance declines for dependent claims. Fine-tuning improves feature completeness and clarity, but significant human revision is still required for legal robustness.

### 28. [Are Large Language Models Good Classifiers? A Study on Edit Intent Classification in Scientific Document Revisions](https://arxiv.org/pdf/2410.02028)

**Summary**: The paper investigates the effectiveness of LLMs in classification tasks, focusing on edit intent classification (EIC) in scientific document revisions. Through extensive experiments and comparisons, the study provides insights into the performance of LLMs for EIC and other classification tasks, and introduces Re3-Sci2.0, a new dataset for empirical analysis of editing behavior.

### 29. [Efficient In-Domain Question Answering for Resource-Constrained Environments](https://arxiv.org/pdf/2409.17648)

**Summary**: The paper introduces CRAFT (Compute-Efficient RAFT), a method that combines Retrieval Augmented Fine Tuning (RAFT) with Low-Rank Adaptation (LoRA) to enhance the efficiency of in-domain question answering in resource-constrained environments. By reducing fine-tuning and storage requirements, CRAFT achieves faster inference times while maintaining performance comparable to traditional RAG setups, making it suitable for knowledge-intensive QA tasks with limited hardware and internet access.

### 30. [Temporally Consistent Factuality Probing for Large Language Models](https://arxiv.org/pdf/2409.14065)

**Summary**: The paper introduces TeCFaP, a new task to evaluate the temporal consistency of factuality in Large Language Models (LLMs), addressing limitations in existing benchmarks. It proposes TEMP-COFAC, a dataset for this task, and extends existing metrics to measure temporal consistency. The study finds that LLMs generally perform poorly on TeCFaP and introduces CoTSeLF, a framework combining multi-task instruction tuning and consistent-time-sensitive reinforcement learning, which significantly improves performance on this task.

### 31. [Pyramid-Driven Alignment: Pyramid Principle Guided Integration of Large Language Models and Knowledge Graphs](https://arxiv.org/pdf/2410.12298)

**Summary**: The paper introduces Pyramid-Driven Alignment (PDA), a framework that integrates Large Language Models (LLMs) with Knowledge Graphs (KGs) to enhance reasoning and reduce hallucinations. PDA employs a hierarchical pyramid structure guided by the Pyramid Principle to better align LLM and KG knowledge, and uses a recursive mechanism to leverage KG reasoning for more accurate question-answering. Experimental results show significant improvements over existing methods, with gains of up to 26.78%.

### 32. [LLM Processes: Numerical Predictive Distributions Conditioned on Natural Language](https://arxiv.org/pdf/2405.12856)

**Summary**: The paper introduces LLM Processes, a method for generating numerical predictive distributions from Large Language Models (LLMs) conditioned on natural language descriptions of prior knowledge. By leveraging LLMs' ability to understand and incorporate expert insights, the authors demonstrate improved predictive performance in various settings, bridging the gap between qualitative descriptions and quantitative predictions.

### 33. [RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models](https://arxiv.org/pdf/2407.05131)

**Summary**: The paper introduces RULE, a method to enhance the factual accuracy of Medical Large Vision Language Models (Med-LVLMs) by addressing challenges in Retrieval-Augmented Generation (RAG). RULE employs a calibrated selection of retrieved contexts to control factuality risk and fine-tunes the model using a preference dataset to balance reliance on inherent knowledge and retrieved contexts. The approach shows a 47.4% average improvement in factual accuracy across medical VQA and report generation tasks.

### 34. [LLM-based Cognitive Models of Students with Misconceptions](https://arxiv.org/pdf/2410.12294)

**Summary**: The paper explores the potential of Large Language Models (LLMs) to simulate student cognition in algebra by accurately replicating misconceptions while still solving problems correctly. Through the development of MalAlgoPy, a Python library that generates authentic student solution patterns, the study finds that LLMs can be instruction-tuned to meet these dual requirements by carefully balancing the ratio of correct to misconception examples in training data, thereby advancing the development of effective AI-driven educational technologies.

### 35. [Understanding the Interplay between Parametric and Contextual Knowledge for Large Language Models](https://arxiv.org/pdf/2410.08414)

**Summary**: The paper explores how LLMs integrate parametric knowledge (PK) with contextual knowledge (CK) and identifies four types of interactions between them. Through the ECHOQA benchmark, it finds that LLMs often suppress PK in favor of CK, even when PK is more relevant, highlighting a vulnerability in their performance on knowledge-intensive tasks.

### 36. [Parameter-Efficient Fine-Tuning of Large Language Models using Semantic Knowledge Tuning](https://arxiv.org/pdf/2410.08598)

**Summary**: The paper introduces Semantic Knowledge Tuning (SK-Tuning), a novel method for fine-tuning Large Language Models (LLMs) using meaningful words instead of random tokens, which enhances performance on tasks like text classification and understanding. SK-Tuning leverages the LLM's zero-shot capabilities to process semantic content, resulting in faster training times, fewer parameters, and superior performance compared to traditional tuning methods.

### 37. [StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Structurization](https://arxiv.org/pdf/2410.08815)

**Summary**: The paper introduces StructRAG, a novel framework that enhances LLMs for knowledge-intensive reasoning tasks by converting raw information into structured formats at inference time. This approach allows for more accurate identification of key information and improved global reasoning, leading to state-of-the-art performance across various challenging tasks.

### 38. [Retriever-and-Memory: Towards Adaptive Note-Enhanced Retrieval-Augmented Generation](https://arxiv.org/pdf/2410.08821)

**Summary**: The paper introduces Adaptive Note-Enhanced Retrieval-Augmented Generation (Adaptive-Note), a novel approach for complex question-answering tasks that addresses the limitations of existing RAG methods by iteratively collecting and integrating new information into an adaptive memory structure. This approach enhances knowledge interactions and employs a note-based strategy to determine when to stop retrieving information, leading to improved answer quality across multiple datasets.

### 39. [NoVo: Norm Voting off Hallucinations with Attention Heads in Large Language Models](https://arxiv.org/pdf/2410.08970)

**Summary**: The paper introduces Norm Voting (NoVo), a lightweight method that leverages attention head norms in Large Language Models to significantly improve factual accuracy in zero-shot multiple-choice questions. NoVo outperforms current state-of-the-art methods by a substantial margin and demonstrates exceptional generalization across diverse datasets, offering new possibilities for enhancing LLM interpretability and robustness.

### 40. [Consecutive Batch Model Editing with HooK Layers](https://arxiv.org/pdf/2403.05330)

**Summary**: The paper introduces CoachHooK, a model editing method designed to efficiently handle both sequential and batch editing scenarios without requiring excessive memory. By utilizing hook layers that maintain a constant size, CoachHooK addresses the limitations of existing methods, demonstrating superior performance in both single-round and consecutive batch editing scenarios. The method's stability over multiple editing steps is also validated through extensive analysis.

### 41. [Evaluating Copyright Takedown Methods for Language Models](https://arxiv.org/pdf/2406.18664)

**Summary**: The paper introduces CoTaEval, an evaluation framework to assess the effectiveness of copyright takedown methods for language models, focusing on their impact on model utility, efficiency, and retention of factual knowledge. The study finds that no single method excels across all metrics, highlighting the need for further research and unresolved challenges in this area.

### 42. [Modular Pluralism: Pluralistic Alignment via Multi-LLM Collaboration](https://arxiv.org/pdf/2406.15951)

**Summary**: The paper introduces Modular Pluralism, a framework that enhances LLMs by integrating specialized, smaller community LMs to better represent diverse human preferences. This modular approach supports three modes of pluralism and is shown to improve performance across various tasks, particularly in handling value-laden and perspective-informed responses. The framework is compatible with both black-box and open-source LLMs, allowing for the addition of new community LMs to address underrepresentation.

### 43. [Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning](https://arxiv.org/pdf/2404.05868)

**Summary**: The paper introduces Negative Preference Optimization (NPO), a novel method for unlearning undesirable data in Large Language Models (LLMs), which addresses the issues of catastrophic collapse and ineffective unlearning seen in gradient ascent-based approaches. NPO demonstrates superior performance in both synthetic and benchmark datasets, achieving significant unlearning while preserving model utility, and producing more coherent outputs compared to existing methods.

### 44. [OAEI-LLM: A Benchmark Dataset for Understanding Large Language Model Hallucinations in Ontology Matching](https://arxiv.org/pdf/2409.14038)

**Summary**: The paper introduces OAEI-LLM, a benchmark dataset designed to assess and understand hallucinations in LLMs when applied to ontology matching (OM) tasks. It extends existing OAEI datasets to include LLM-specific hallucinations, providing a framework for evaluating and mitigating these issues in OM.

### 45. [Unraveling Cross-Modality Knowledge Conflicts in Large Vision-Language Models](https://arxiv.org/pdf/2410.03659)

**Summary**: The paper addresses the issue of cross-modality parametric knowledge conflicts in Large Vision-Language Models (LVLMs), where inconsistencies between visual and textual components lead to errors. The authors introduce a systematic approach to detect and mitigate these conflicts, including a dynamic contrastive decoding method that improves model accuracy by 2.24% on average using LLaVA-34B.

### 46. [MKGL: Mastery of a Three-Word Language](https://arxiv.org/pdf/2410.07526)

**Summary**: The paper introduces MKGL, a specialized Knowledge Graph Language (KGL) designed to integrate LLMs with knowledge graphs (KGs). By structuring sentences as entity-relation-entity triplets, MKGL minimizes hallucinations and enhances accuracy. The study demonstrates that LLMs can achieve fluency in KGL through tailored learning methods, significantly outperforming traditional KG embedding techniques in KG completion tasks.

### 47. [KRAG Framework for Enhancing LLMs in the Legal Domain](https://arxiv.org/pdf/2410.07551)

**Summary**: The paper introduces the Knowledge Representation Augmented Generation (KRAG) framework, which enhances Large Language Models (LLMs) by incorporating critical knowledge entities and relationships often missing in standard datasets. Specifically, in the legal domain, KRAG is implemented through Soft PROLEG, which uses inference graphs to improve LLMs' ability to provide structured legal reasoning and explanations. This approach significantly advances natural language understanding in specialized domains like law.

### 48. [Uncovering Overfitting in Large Language Model Editing](https://arxiv.org/pdf/2410.07819)

**Summary**: The paper identifies a phenomenon called Editing Overfit in Large Language Models, where edited models overly prioritize the edit target, leading to poor generalization in complex tasks. To address this, the authors introduce a new benchmark, EVOKE, and propose a strategy called Learn to Inference (LTI) with a Multi-stage Inference Constraint module, which helps edited models recall knowledge more effectively, reducing overfitting.

### 49. [Fine-Tuning Language Models for Ethical Ambiguity: A Comparative Study of Alignment with Human Responses](https://arxiv.org/pdf/2410.07826)

**Summary**: The study investigates the alignment of language models with human judgments in morally ambiguous scenarios by fine-tuning models on curated datasets from the Scruples project. Significant improvements in model performance were observed post-fine-tuning, particularly in cross-entropy and Dirichlet scores, highlighting the potential for enhancing ethical reasoning in language models. However, the fine-tuned models still lagged behind BERT and RoBERTa in cross-entropy scores, suggesting ongoing challenges in capturing human judgment nuances.

### 50. [Disease Entity Recognition and Normalization is Improved with Large Language Model Derived Synthetic Normalized Mentions](https://arxiv.org/pdf/2410.07951)

**Summary**: The study demonstrates that fine-tuning a Large Language Model (LLM) to generate synthetic normalized mentions of disease entities significantly improves Disease Entity Normalization (DEN) performance, particularly for out-of-distribution data, while yielding only modest enhancements for Disease Entity Recognition (DER). The synthetic data augmentation led to substantial accuracy improvements in DEN across multiple datasets, suggesting that LLM-generated data can effectively supplement traditional training methods.

### 51. [A Closer Look at Machine Unlearning for Large Language Models](https://arxiv.org/pdf/2410.08109)

**Summary**: The paper examines the challenges of machine unlearning in LLMs, particularly in removing specific content while maintaining model performance. It introduces new evaluation metrics and proposes methods like maximizing entropy for untargeted unlearning and answer preservation loss for targeted unlearning to address these challenges. Experimental results show the effectiveness of these approaches across various unlearning scenarios.

### 52. [Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study over Open-ended Question Answering](https://arxiv.org/pdf/2410.08085)

**Summary**: The paper introduces OKGQA, a new benchmark for evaluating the effectiveness of Knowledge Graphs (KGs) in enhancing Large Language Models (LLMs) in open-ended question answering scenarios. The benchmark aims to assess both the reduction in hallucinations and the improvement in reasoning capabilities of LLMs when integrated with KGs, and it also includes a perturbed version (OKGQA-P) to test model robustness against KG errors. The study seeks to determine whether KGs can make LLMs more trustworthy in real-world applications and to guide future research in this area.

### 53. [Insight Over Sight? Exploring the Vision-Knowledge Conflicts in Multimodal LLMs](https://arxiv.org/pdf/2410.08145)

**Summary**: The paper investigates the issue of vision-knowledge conflicts in Multimodal Large Language Models (MLLMs), where visual information contradicts the model's internal commonsense knowledge. It introduces a benchmark with 374 images and 1,122 QA pairs to assess conflict resolution, finding that models often over-rely on textual queries. A new prompting strategy, "Focus-on-Vision" (FoV), is proposed to improve models' reliance on visual data, enhancing their conflict-resolution capabilities.

### 54. [From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions](https://arxiv.org/pdf/2410.08197)

**Summary**: The paper introduces DRAFT, a framework that dynamically refines tool documentation for Large Language Models (LLMs) by analyzing feedback and interaction trails. Through a trial-and-error approach involving experience gathering, learning, and documentation rewriting, DRAFT iteratively improves documentation quality, enhancing LLMs' understanding and use of tools. The framework's effectiveness is demonstrated through experiments showing improved documentation quality and cross-model generalization.

### 55. [Mitigating Gender Bias in Code Large Language Models via Model Editing](https://arxiv.org/pdf/2410.07820)

**Summary**: The paper introduces CodeGenBias, a dataset and FB-Score metric to evaluate gender bias in code Large Language Models (LLMs), and proposes MG-Editing, a multi-granularity model editing approach to mitigate this bias. Experiments show that MG-Editing effectively reduces gender bias while preserving code generation capabilities, with the best performance at row and neuron levels of granularity.

### 56. [Composite Learning Units: Generalized Learning Beyond Parameter Updates to Transform LLMs into Adaptive Reasoners](https://arxiv.org/pdf/2410.08037)

**Summary**: The paper introduces Composite Learning Units (CLUs) to enhance Large Language Models (LLMs) with continuous, generalized learning capabilities without traditional parameter updates. CLUs utilize a dynamic knowledge repository, including a General Knowledge Space and a Prompt-Specific Knowledge Space, to iteratively refine reasoning through goal-driven interactions and feedback, enabling adaptive reasoning and autonomous learning from past experiences.

### 57. [SLIM: Let LLM Learn More and Forget Less with Soft LoRA and Identity Mixture](https://arxiv.org/pdf/2410.07739)

**Summary**: The paper introduces SLIM, a novel mixture of expert framework that combines Soft LoRA and Identity Mixture to efficiently fine-tune LLMs while mitigating catastrophic forgetting. SLIM dynamically routes between LoRA adapters and skipping connections, enhancing out-of-domain distinction and preserving the base model's general capabilities. Experimental results show that SLIM outperforms existing parameter-efficient fine-tuning methods in balancing downstream task performance and preventing forgetting.

### 58. [Fuse to Forget: Bias Reduction and Selective Memorization through Model Fusion](https://arxiv.org/pdf/2311.07682)

**Summary**: The paper explores the use of model fusion to reduce unwanted knowledge, such as shortcuts, social biases, and memorization of training data, in fine-tuned language models. Through experiments, it shows that model fusion enhances shared knowledge while forgetting unshared knowledge, making it a promising tool for debiasing and addressing privacy concerns in language models.

### 59. [AKEW: Assessing Knowledge Editing in the Wild](https://arxiv.org/pdf/2402.18909)

**Summary**: The paper introduces AKEW, a benchmark for assessing knowledge editing in practical scenarios, addressing the limitations of current evaluations that focus on structured facts from curated datasets. AKEW includes settings for structured facts, unstructured texts, and extracted triplets, and features both counterfactual and real-world knowledge updates, revealing significant gaps between existing methods and practical needs.

### 60. [AutoRD: An Automatic and End-to-End System for Rare Disease Knowledge Graph Construction Based on Ontologies-enhanced Large Language Models](https://arxiv.org/pdf/2403.00953)

**Summary**: The paper introduces AutoRD, an end-to-end system designed to automate the construction of a knowledge graph for rare diseases by extracting entities and their relations from medical texts. Leveraging ontologies-enhanced Large Language Models, AutoRD integrates up-to-date structured knowledge and outperforms traditional methods and common LLMs in rare disease extraction tasks.

### 61. [LLaMP: Large Language Model Made Powerful for High-fidelity Materials Knowledge Retrieval and Distillation](https://arxiv.org/pdf/2401.17244)

**Summary**: The paper introduces LLaMP, a multimodal retrieval-augmented generation framework that enhances Large Language Models (LLMs) for high-fidelity materials knowledge retrieval and distillation. By integrating hierarchical reasoning-and-acting agents with computational and experimental data, LLaMP effectively reduces hallucination and biases in LLMs, demonstrating strong tool usage and self-consistency in handling complex materials science tasks. The framework also showcases capabilities in editing crystal structures and running molecular dynamics simulations, offering a nearly hallucination-free approach to materials informatics.

### 62. [Unlocking the Power of Large Language Models for Entity Alignment](https://arxiv.org/pdf/2402.15048)

**Summary**: The paper introduces ChatEA, a novel framework that leverages LLMs to enhance Entity Alignment (EA) in knowledge graphs. By translating KG structures into a format understandable by LLMs and employing a two-stage EA strategy, ChatEA improves EA accuracy by utilizing LLMs' extensive background knowledge and multi-step reasoning capabilities. Experimental results demonstrate ChatEA's superior performance over traditional methods.

### 63. [Verification and Refinement of Natural Language Explanations through LLM-Symbolic Theorem Proving](https://arxiv.org/pdf/2405.01379)

**Summary**: The paper introduces Explanation-Refiner, a neuro-symbolic framework that integrates Large Language Models (LLMs) with Theorem Provers (TPs) to verify and refine natural language explanations for Natural Language Inference (NLI). By leveraging TPs for formal validation and feedback, the framework aims to enhance the quality and logical correctness of explanations across various domains, thereby addressing the limitations of current crowd-sourced validation methods.

### 64. [ClaimBrush: A Novel Framework for Automated Patent Claim Refinement Based on Large Language Models](https://arxiv.org/pdf/2410.05575)

**Summary**: The paper introduces ClaimBrush, a framework for automated patent claim refinement using large language models. It includes a dataset of actual patent claim rewriting cases and a fine-tuned rewriting model, enhanced by preference optimization based on patent examiners' feedback. Experimental results demonstrate superior performance over heuristic baselines and zero-shot learning methods.

### 65. [Dr-LLaVA: Visual Instruction Tuning with Symbolic Clinical Grounding](https://arxiv.org/pdf/2405.19567)

**Summary**: The paper introduces Dr-LLaVA, a Vision-Language Model (VLM) designed for medical applications, which uses a novel alignment algorithm to ground its outputs in clinical reasoning. This algorithm leverages symbolic representations of medical knowledge to generate training data and create an automatic reward function, eliminating the need for human involvement and reducing costs. Dr-LLaVA demonstrates strong performance in multi-turn medical conversations, particularly in analyzing bone marrow pathology slides.

### 66. [Can Large Language Models Understand DL-Lite Ontologies? An Empirical Study](https://arxiv.org/pdf/2406.17532)

**Summary**: The paper investigates whether LLMs can understand DL-Lite ontologies, focusing on six representative tasks from syntactic and semantic perspectives. The study reveals that LLMs effectively grasp formal syntax and model-theoretic semantics but face challenges with TBox NI transitivity and large ABoxes. The findings aim to inform future knowledge engineering solutions.

### 67. [Revisiting the Superficial Alignment Hypothesis](https://arxiv.org/pdf/2410.03717)

**Summary**: The paper challenges the Superficial Alignment Hypothesis by demonstrating that post-training with increasing fine-tuning examples significantly improves language model performance across various tasks, including mathematical reasoning and multihop reasoning. The study reveals that model performance scales as a power law with the number of fine-tuning examples, indicating that post-training is crucial for enhancing reasoning abilities and integrating new knowledge, contrary to the hypothesis's claims.

### 68. [Neurosymbolic AI approach to Attribution in Large Language Models](https://arxiv.org/pdf/2410.03726)

**Summary**: The paper proposes a Neurosymbolic AI (NesyAI) approach to improve attribution in LLMs, addressing issues like hallucinations, biases, and unreliable sources by combining neural networks with structured symbolic reasoning. This integration aims to provide more reliable, interpretable, and adaptable systems for ensuring the factual accuracy and reliability of LLM outputs.

### 69. [Reasoning Elicitation in Language Models via Counterfactual Feedback](https://arxiv.org/pdf/2410.03767)

**Summary**: The paper introduces new metrics to evaluate the reasoning capabilities of language models, particularly in counterfactual scenarios, and proposes fine-tuning methods to enhance these abilities. The study evaluates the fine-tuned models across various reasoning tasks, demonstrating improved generalization and performance compared to baseline models.

### 70. [Precision Knowledge Editing: Enhancing Safety in Large Language Models](https://arxiv.org/pdf/2410.03772)

**Summary**: The paper introduces Precision Knowledge Editing (PKE), a technique that enhances the safety of LLMs by more effectively identifying and modifying toxic parameter regions. PKE, which builds on existing knowledge editing methods, uses neuron weight tracking and activation pathway tracing to achieve finer granularity in managing toxic content. Experiments show that PKE significantly reduces the attack success rate across various models while maintaining overall performance, outperforming closed-source models in terms of safety.

### 71. [Neuron-Level Sequential Editing for Large Language Models](https://arxiv.org/pdf/2410.04045)

**Summary**: The paper introduces Neuron-Level Sequential Editing (NSE), a novel method for continuously updating LLMs through multi-round editing without requiring costly retraining. NSE optimizes hidden states using original weights to prevent model failure and iteratively selects neurons for editing to mitigate forgetting, outperforming existing methods in sequential model editing tasks.

### 72. [DiDOTS: Knowledge Distillation from Large-Language-Models for Dementia Obfuscation in Transcribed Speech](https://arxiv.org/pdf/2410.04188)

**Summary**: The paper introduces DiDOTS, a method that leverages Large-Language-Models (LLMs) to obfuscate dementia indicators in speech transcripts, addressing privacy concerns without relying on large labeled datasets. DiDOTS uses knowledge distillation to create a more efficient model with significantly fewer parameters, outperforming existing methods in both privacy and utility preservation.

### 73. [Persona Knowledge-Aligned Prompt Tuning Method for Online Debate](https://arxiv.org/pdf/2410.04239)

**Summary**: The paper introduces a novel framework that leverages ChatGPT's capabilities to simulate audience personas and enhance argument quality assessment by aligning persona knowledge with smaller language models through prompt tuning. This approach significantly improves performance in debate scenarios by integrating audience-specific characteristics, marking a first in combining argument persuasiveness with audience personae.

### 74. [Toxic Subword Pruning for Dialogue Response Generation on Large Language Models](https://arxiv.org/pdf/2410.04155)

**Summary**: The paper introduces ToxPrune, a novel algorithm that prunes toxic subwords from Byte Pair Encoding (BPE) in trained Large Language Models (LLMs) to prevent the generation of toxic content. Contrary to previous findings that pruning BPE tokens can harm machine translation tasks, the authors demonstrate that ToxPrune effectively reduces toxicity in dialogue response generation and even improves the diversity of responses in models like NSFW-3B and Llama-3.1-6B.

### 75. [Entity Insertion in Multilingual Linked Corpora: The Case of Wikipedia](https://arxiv.org/pdf/2410.04254)

**Summary**: The paper introduces the task of entity insertion in information networks, focusing on Wikipedia, and addresses the challenge of locating suitable positions for new links in text without predefined anchors. The authors develop LocEI and its multilingual variant XLocEI, which outperform baseline models, including GPT-4, and demonstrate effective zero-shot performance across languages. This work is crucial for enhancing link insertion across Wikipedia's multilingual editions.

### 76. [Mechanistic Behavior Editing of Language Models](https://arxiv.org/pdf/2410.04277)

**Summary**: The paper introduces TaRot, a method for task adaptation in LLMs that uses learnable rotation matrices optimized via Bayesian Optimization to improve performance on classification and generation tasks. TaRot enhances both zero-shot and few-shot performance, with average improvements of 23.81% and 11.15% respectively across various models and tasks.

### 77. [Collapsed Language Models Promote Fairness](https://arxiv.org/pdf/2410.04472)

**Summary**: The paper investigates the relationship between Neural Collapse and fairness in language models, finding that debiased models exhibit collapsed alignment between token representations and word embeddings. This insight leads to a new fine-tuning method that enhances fairness across various debiasing techniques while maintaining performance on standard language understanding tasks.

### 78. [Wrong-of-Thought: An Integrated Reasoning Framework with Multi-Perspective Verification and Wrong Information](https://arxiv.org/pdf/2410.04463)

**Summary**: The paper introduces Wrong-of-Thought (WoT), an integrated reasoning framework designed to improve the performance of Large Language Models (LLMs) by addressing two key issues: the reliance on single verification methods and the ignorance of wrong information during reasoning. WoT incorporates multi-perspective verification to refine reasoning processes and utilizes wrong information to prevent recurring errors. Experimental results show that WoT outperforms existing methods across multiple datasets and LLMs, particularly in challenging computation tasks.

### 79. [Formality is Favored: Unraveling the Learning Preferences of Large Language Models on Data with Conflicting Knowledge](https://arxiv.org/pdf/2410.04784)

**Summary**: The study investigates how LLMs handle conflicting information in training data, finding that they prefer formal texts and those with fewer spelling errors, similar to human preferences. This preference leads to faster learning and better treatment of knowledge in data with these features, especially in larger models, and can be influenced by manipulating data consistency.

### 80. [Document-level Causal Relation Extraction with Knowledge-guided Binary Question Answering](https://arxiv.org/pdf/2410.04752)

**Summary**: The paper introduces a Knowledge-guided binary Question Answering (KnowQA) method for Event-Event Causal Relation Extraction (ECRE), addressing challenges like lack of document-level modeling and causal hallucinations. The proposed method, involving Event Structure Construction and Binary Question Answering, achieves state-of-the-art performance on the MECI dataset and demonstrates high generalizability and low inconsistency, especially with complete event structures post-fine-tuning.

### 81. [Activation Scaling for Steering and Interpreting Language Models](https://arxiv.org/pdf/2410.04962)

**Summary**: The paper explores the concept of steering language models by scaling activation vectors to correct incorrect predictions, such as flipping "Rome is in France" to "Rome is in Italy." The authors propose a three-term objective for effective, faithful, and minimal interventions, achieved through gradient-based optimization. They demonstrate that activation scaling is more minimal and interpretable than steering vectors, allowing for precise identification of model components.

### 82. [ZEBRA: Zero-Shot Example-Based Retrieval Augmentation for Commonsense Question Answering](https://arxiv.org/pdf/2410.05077)

**Summary**: The paper introduces ZEBRA, a zero-shot question answering framework that enhances commonsense reasoning by combining retrieval, case-based reasoning, and introspection without requiring additional training of the language model. ZEBRA outperforms existing methods and strong language models, achieving an average accuracy improvement of up to 4.5 points across eight benchmarks.

### 83. [Deciphering the Interplay of Parametric and Non-parametric Memory in Retrieval-augmented Language Models](https://arxiv.org/pdf/2410.05162)

**Summary**: The study investigates how Retrieval-Augmented Generation (RAG) models like \textsc{Atlas} balance parametric (internal) and non-parametric (retrieved) memory during response generation. Through causal mediation analysis, it reveals that the model prioritizes retrieved context over internal knowledge when both are available, and identifies two key mechanisms: determining context relevance and computing output representations for copying relevant information.

### 84. [Metadata Matters for Time Series: Informative Forecasting with Transformers](https://arxiv.org/pdf/2410.03806)

**Summary**: The paper introduces MetaTST, a novel approach that integrates metadata into Transformer models for time series forecasting, enhancing accuracy and interpretability. By converting unstructured metadata into structured text and encoding it with large language models, MetaTST enriches the embedding process and improves forecasting performance across diverse scenarios, achieving state-of-the-art results in both short- and long-term benchmarks.

### 85. [A Pluggable Common Sense-Enhanced Framework for Knowledge Graph Completion](https://arxiv.org/pdf/2410.04488)

**Summary**: The paper introduces a pluggable framework for knowledge graph completion (KGC) that integrates both factual and common sense information to improve inference accuracy. The framework is adaptable to different KGs, automatically generating common sense from factual triples and employing techniques like common sense-guided negative sampling and a dual scoring scheme. It demonstrates superior performance and scalability across various KGC tasks compared to existing models.

### 86. [Augmenting Black-box LLMs with Medical Textbooks for Biomedical Question Answering (Published in Findings of EMNLP 2024)](https://arxiv.org/pdf/2309.02233)

**Summary**: The paper introduces LLM-AMT, a system that enhances LLMs like ChatGPT with medical textbooks to improve their performance in biomedical question answering. By integrating authoritative medical knowledge through specialized modules, LLM-AMT significantly boosts response accuracy, outperforming even specialized medical models like Med-PaLM 2. The study highlights that medical textbooks are more effective than Wikipedia as a knowledge source in this domain.

### 87. [Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue](https://arxiv.org/pdf/2401.04700)

**Summary**: The paper investigates the side effects of model editing on LLMs, finding that while editing improves factuality, it often degrades general abilities like reasoning and question answering. To address this, the authors propose RECT, a regularization method that constrains the complexity of weight updates, effectively mitigating side effects while preserving editing performance.

### 88. [Corrective Retrieval Augmented Generation](https://arxiv.org/pdf/2401.15884)

**Summary**: The paper introduces Corrective Retrieval Augmented Generation (CRAG), a method designed to enhance the robustness of retrieval-augmented generation (RAG) by incorporating a retrieval evaluator to assess the quality of retrieved documents and trigger appropriate actions. CRAG also employs large-scale web searches to augment retrieval results and uses a decompose-then-recompose algorithm to focus on key information, improving performance across various RAG-based approaches in both short- and long-form generation tasks.

### 89. [Editing Conceptual Knowledge for Large Language Models](https://arxiv.org/pdf/2403.06259)

**Summary**: The paper introduces a novel approach to editing conceptual knowledge in Large Language Models (LLMs) by creating the ConceptEdit benchmark dataset and new evaluation metrics. It finds that while existing editing methods can modify concept-level definitions, they often distort related instantial knowledge, highlighting the need for improved techniques to balance these changes.

### 90. [FAC$^2$E: Better Understanding Large Language Model Capabilities by Dissociating Language and Cognition](https://arxiv.org/pdf/2403.00126)

**Summary**: The paper introduces FAC$^2$E, a framework for evaluating LLMs by distinguishing between language and cognitive capabilities. It breaks down the evaluation process into three sub-steps: knowledge recall, knowledge utilization, and problem-solving, providing a detailed diagnosis of LLMs' performance. The study identifies a common weakness in knowledge utilization and suggests a knowledge-enhanced method to improve LLM performance.

### 91. [Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems](https://arxiv.org/pdf/2402.17840)

**Summary**: The paper investigates the vulnerability of Retrieval-Augmented Generation (RAG) systems, particularly those using instruction-tuned Language Models (LMs), to datastore leakage through prompt injection. The study reveals that adversaries can exploit these systems to extract verbatim text data from the datastore, with the risk increasing as model size scales up. The authors demonstrate successful extraction from various LMs and propose mitigation strategies, including position bias elimination, to reduce this vulnerability.

### 92. [Red Teaming Language Models for Processing Contradictory Dialogues](https://arxiv.org/pdf/2405.10128)

**Summary**: The paper introduces a novel task for processing contradictory dialogues in language models, aiming to detect and modify self-contradictory statements. A Red Teaming framework is developed using a dataset of contradictory dialogues with explanatory labels, which improves detection, explanation, and modification of contradictions. The study underscores the significance of addressing logical inconsistencies in conversational AI.

### 93. [Student Data Paradox and Curious Case of Single Student-Tutor Model: Regressive Side Effects of Training LLMs for Personalized Learning](https://arxiv.org/pdf/2404.15156)

**Summary**: The paper identifies the "Student Data Paradox," where training Large Language Models (LLMs) on extensive student-tutor dialogue datasets to personalize education leads to a decline in the models' factual knowledge and reasoning abilities. The study demonstrates this paradox through quantitative analysis and introduces "hallucination tokens" as a partial solution, highlighting the ongoing challenge of balancing accurate student behavior modeling with maintaining the LLM's educational integrity.

### 94. [Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete Knowledge Graph Question Answering](https://arxiv.org/pdf/2404.14741)

**Summary**: The paper introduces Generate-on-Graph (GoG), a training-free method that leverages Large Language Models (LLMs) to address Incomplete Knowledge Graph Question Answering (IKGQA) by generating new factual triples when the provided Knowledge Graph (KG) is insufficient. GoG operates through a Thinking-Searching-Generating framework, treating LLMs as both agents and KGs, and outperforms previous methods in experimental evaluations on two datasets.

### 95. [Exploring the Compositional Deficiency of Large Language Models in Mathematical Reasoning](https://arxiv.org/pdf/2405.06680)

**Summary**: The paper investigates the compositionality of LLMs in mathematical reasoning by introducing logical traps into problem descriptions, creating a dataset called MathTrap. The study finds that while LLMs possess the necessary mathematical knowledge, they fail to spontaneously combine it with knowledge of logical traps to solve novel cases. Performance can be passively improved through external interventions like natural language prompts and fine-tuning, but systematic compositionality remains a challenge for LLMs.

### 96. [Representation noising effectively prevents harmful fine-tuning on LLMs](https://arxiv.org/pdf/2405.14577)

**Summary**: The paper introduces Representation Noising (RepNoise), a defense mechanism against harmful fine-tuning attacks on LLMs. RepNoise removes information about harmful representations, making it difficult for attackers to recover them during fine-tuning, even when they have access to the model weights. The method is shown to generalize across different types of harm and does not impair the model's performance on harmless tasks.

### 97. [WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models](https://arxiv.org/pdf/2405.14768)

**Summary**: The paper introduces WISE, a novel approach for lifelong model editing in LLMs that addresses the challenge of balancing reliability, generalization, and locality when updating knowledge. WISE employs a dual parametric memory scheme, with a main memory for pretrained knowledge and a side memory for edited knowledge, along with a router to manage queries. A knowledge-sharding mechanism ensures that edits are stored in distinct subspaces, preventing conflicts and enabling effective merging into a shared memory. Experiments demonstrate WISE's superior performance in various LLM architectures, including GPT, LLaMA, and Mistral, across different settings.

### 98. [RLSF: Reinforcement Learning via Symbolic Feedback](https://arxiv.org/pdf/2405.16661)

**Summary**: The paper introduces Reinforcement Learning via Symbolic Feedback (RLSF), a novel fine-tuning method for Large Language Models (LLMs) that leverages reasoning tools to provide detailed, token-level feedback through poly-sized certificates, addressing limitations of traditional RLHF methods. RLSF-based fine-tuning demonstrates superior performance across various tasks, enabling smaller LLMs to outperform larger, closed-source models like GPT-4.

### 99. [Self-MoE: Towards Compositional Large Language Models with Self-Specialized Experts](https://arxiv.org/pdf/2406.12034)

**Summary**: The paper introduces Self-MoE, a method that converts monolithic LLMs into modular systems of self-specialized experts (MiXSE) using self-generated synthetic data. This approach enhances the LLM's performance across various tasks without requiring extensive human-labeled data, showing significant improvements (6.5% on average) in benchmarks and outperforming other methods in flexibility and interpretability. The study underscores the importance of modularity and the potential for self-improvement in creating efficient and adaptable systems.

### 100. [DogeRM: Equipping Reward Models with Domain Knowledge through Model Merging](https://arxiv.org/pdf/2407.01470)

**Summary**: The paper introduces DogeRM, a framework that merges domain-specific knowledge into general reward models to improve performance in reinforcement learning from human feedback (RLHF). By integrating expert annotations through model merging, DogeRM reduces the need for costly preference data collection and enhances model alignment across various benchmarks.



---

*Last updated on 2024-10-18*