# Awesome-LLM-Model-Editing

## Awesome LLM Model Editing

Welcome to the **Awesome LLM Model Editing** repository! This project curates a list of high-quality resources related to LLM Model Editing, including research papers, tools, libraries, and more.

## Daily Updates

### 2024-10-26

### 1. [CorrectionLM: Self-Corrections with SLM for Dialogue State Tracking](https://arxiv.org/pdf/2410.18209)

**Summary**: The paper introduces CORRECTIONLM, a novel framework that allows small language models (SLMs) to self-correct using in-context exemplars, eliminating the need for LLMs. Applied to dialogue state tracking tasks in low-resource settings, CORRECTIONLM achieves performance comparable to state-of-the-art LLMs while significantly reducing computational costs.

### 2. [Towards Understanding the Fragility of Multilingual LLMs against Fine-Tuning Attacks](https://arxiv.org/pdf/2410.18210)

**Summary**: The paper investigates the vulnerability of multilingual Large Language Models (LLMs) to fine-tuning attacks, finding that safety alignment can be compromised across languages using a few adversarial examples. It introduces Safety Information Localization (SIL) to identify safety-related parameters, revealing that altering just 20% of these parameters can break safety alignment globally. The study also supports the alternative pathways hypothesis, suggesting that freezing safety parameters does not fully prevent attacks.

### 3. [Improving Model Factuality with Fine-grained Critique-based Evaluator](https://arxiv.org/pdf/2410.18359)

**Summary**: The paper introduces FenCE, a fine-grained critique-based factuality evaluator designed to improve the factual accuracy of language models. By augmenting training data with textual critiques and claim-level judgments, FenCE enhances the evaluator's accuracy and is used to refine model responses, leading to a significant improvement in factuality rates. The approach outperforms existing finetuning methods, demonstrating its effectiveness in guiding the development of more factual models.

### 4. [From Imitation to Introspection: Probing Self-Consciousness in Language Models](https://arxiv.org/pdf/2410.18819)

**Summary**: The paper investigates the potential for self-consciousness in advanced language models by defining ten core concepts related to self-consciousness and using causal structural games to explore their representation within models. The study concludes that while models exhibit early signs of self-consciousness, positive manipulation of these concepts is challenging, though they can be enhanced through fine-tuning.

### 5. [Delving into the Reversal Curse: How Far Can Large Language Models Generalize?](https://arxiv.org/pdf/2410.18808)

**Summary**: The paper investigates the "reversal curse" in LLMs, where models struggle to infer "B is A" from "A is B". It finds that LLMs can generalize this reversal in certain contexts, particularly when the structure of the fact aligns with training data. The study reveals an inherent bias in LLMs' fact recall, emphasizing the importance of document structure and suggesting that training alone may not mitigate this bias, offering insights for improving LLM learning methods.

### 6. [We Augmented Whisper With kNN and You Won't Believe What Came Next](https://arxiv.org/pdf/2410.18850)

**Summary**: The paper explores the integration of k-nearest neighbor (kNN) search into the Whisper speech recognition model to enhance performance across various languages, domains, and speaker characteristics without causing catastrophic forgetting. The study demonstrates that kNN adaptation significantly improves recognition accuracy, particularly for speakers with distinct accents, gender, and age groups.

### 7. [WAFFLE: Multi-Modal Model for Automated Front-End Development](https://arxiv.org/pdf/2410.18362)

**Summary**: The paper introduces WAFFLE, a multi-modal model designed to automate front-end development by addressing the challenges of representing HTML's hierarchical structure and aligning UI designs with HTML code. WAFFLE employs a structure-aware attention mechanism and contrastive fine-tuning to enhance Large Language Models' capabilities, achieving significant improvements in HTML match, visual similarity, and other metrics on benchmark tests.

### 8. [Demystifying Large Language Models for Medicine: A Primer](https://arxiv.org/pdf/2410.18856)

**Summary**: The paper "Demystifying Large Language Models for Medicine: A Primer" provides a comprehensive guide for healthcare professionals to effectively integrate LLMs into clinical practice. It outlines a structured methodology, including task formulation, model selection, prompt engineering, fine-tuning, and deployment considerations, to ensure the safe and impactful use of LLMs in medical tasks such as clinical documentation and patient matching to clinical trials. The primer emphasizes the importance of regulatory compliance, ethical guidelines, and continuous monitoring to address potential biases and fairness issues.

### 9. [RET-LLM: Towards a General Read-Write Memory for Large Language Models](https://arxiv.org/pdf/2305.14322)

**Summary**: The paper introduces RET-LLM, a framework that enhances LLMs with a general read-write memory unit, enabling them to store and retrieve knowledge in the form of triplets for improved task performance. The memory unit is designed to be scalable, aggregatable, and interpretable, and it demonstrates superior performance in question answering tasks, particularly in handling temporal-based queries.

### 10. [Zero-shot Persuasive Chatbots with LLM-Generated Strategies and Information Retrieval](https://arxiv.org/pdf/2407.03585)

**Summary**: The paper introduces PersuaBot, a zero-shot persuasive chatbot that leverages Large Language Models (LLMs) to generate nuanced persuasion strategies and integrates information retrieval to ensure factual accuracy. The chatbot was tested in three diverse domains—donation solicitation, recommendations, and health intervention—demonstrating improved persuasiveness and factual correctness compared to existing methods.

### 11. [The Factuality Tax of Diversity-Intervened Text-to-Image Generation: Benchmark and Fact-Augmented Intervention](https://arxiv.org/pdf/2407.00377)

**Summary**: The paper introduces DoFaiR, a benchmark to assess the trade-off between diversity interventions and demographic factuality in Text-to-Image (T2I) models, revealing that diversity prompts often lead to historically inaccurate representations. To address this, the authors propose Fact-Augmented Intervention (FAI), which uses Large Language Models to incorporate historical facts into the generation process, thereby enhancing factuality while maintaining diversity.

### 12. [MoESD: Mixture of Experts Stable Diffusion to Mitigate Gender Bias](https://arxiv.org/pdf/2407.11002)

**Summary**: The paper introduces MoESD, a Mixture of Experts Stable Diffusion model with Bias Adapters (BiAs) to address gender bias in text-to-image generation. By identifying and mitigating bias in the text encoder's latent space, the authors demonstrate successful reduction of gender bias while preserving image quality, emphasizing the importance of a special token in the prompt for effective bias mitigation.

### 13. [PortLLM: Personalizing Evolving Large Language Models with Training-Free and Portable Model Patches](https://arxiv.org/pdf/2410.10870)

**Summary**: The paper introduces PortLLM, a training-free framework for personalizing evolving LLMs by creating lightweight model update patches that capture domain-specific knowledge and can be seamlessly integrated into updated LLMs. The framework demonstrates significant reductions in GPU memory usage while maintaining performance comparable to traditional fine-tuning methods, offering a practical solution for users with limited resources.

### 14. [KatzBot: Revolutionizing Academic Chatbot for Enhanced Communication](https://arxiv.org/pdf/2410.16385)

**Summary**: The paper introduces KatzBot, an advanced academic chatbot utilizing KatzGPT, a custom Large Language Model fine-tuned on university-specific data. KatzBot demonstrates superior accuracy and domain relevance compared to existing open-source LLMs, significantly improving user satisfaction in university communication.

### 15. [To the Globe (TTG): Towards Language-Driven Guaranteed Travel Planning](https://arxiv.org/pdf/2410.16456)

**Summary**: The paper introduces To the Globe (TTG), a real-time system that uses a fine-tuned Large Language Model to translate natural language travel requests into symbolic form, enabling the generation of optimal itineraries through Mixed Integer Linear Programming solvers. TTG achieves a 91% exact match in translating user requests and consistently high user satisfaction, with a processing time of approximately 5 seconds per request.

### 16. [Does your LLM truly unlearn? An embarrassingly simple approach to recover unlearned knowledge](https://arxiv.org/pdf/2410.16454)

**Summary**: The paper investigates whether machine unlearning in LLMs truly erases unwanted knowledge or merely hides it, as current benchmarks fail to detect this. The study finds that applying quantization to unlearned models can restore the "forgotten" information, with unlearned models retaining up to 83% of the intended forgotten knowledge after 4-bit quantization. The authors propose a quantization-robust unlearning strategy to address this issue.

### 17. [Distill-SynthKG: Distilling Knowledge Graph Synthesis Workflow for Improved Coverage and Efficiency](https://arxiv.org/pdf/2410.16597)

**Summary**: The paper introduces Distill-SynthKG, a novel approach to efficiently synthesize knowledge graphs (KGs) from LLMs by distilling a multi-step workflow into a single-step process, significantly reducing inference calls. The method outperforms larger baseline models in KG quality and excels in retrieval and question-answering tasks, supported by a new graph-based retrieval framework and repurposed evaluation datasets.

### 18. [Controlled Low-Rank Adaptation with Subspace Regularization for Continued Training on Large Language Models](https://arxiv.org/pdf/2410.16801)

**Summary**: The paper introduces Controlled LoRA (CLoRA), a subspace regularization method for low-rank adaptation (LoRA) in LLMs to mitigate catastrophic forgetting during continued training. CLoRA constrains updates to the null space of the updating matrix, achieving superior performance in both in-domain and out-domain evaluations compared to existing LoRA methods, while effectively balancing model capacity and forgetting.

### 19. [Learning Mathematical Rules with Large Language Models](https://arxiv.org/pdf/2410.16973)

**Summary**: The paper investigates the capability of large language models to learn and apply specific mathematical rules, such as distributivity and equation simplification, through empirical analysis. By fine-tuning models on synthetic data generated with rigorous methodology, the study demonstrates that these models can generalize and reuse the learned rules effectively in word problem contexts.

### 20. [Atomic Fact Decomposition Helps Attributed Question Answering](https://arxiv.org/pdf/2410.16708)

**Summary**: The paper introduces an Atomic Fact Decomposition-based Retrieval and Editing (ARE) framework to improve Attributed Question Answering (AQA) by decomposing long-form answers into atomic facts using instruction-tuned Large Language Models (LLMs). The framework retrieves evidence for these facts and edits them as needed, ensuring more accurate and reliable answers. The method outperforms existing approaches and introduces a new metric, $Attr_{p}$, to evaluate evidence attribution precision.

### 21. [Trustworthy Alignment of Retrieval-Augmented Large Language Models via Reinforcement Learning](https://arxiv.org/pdf/2410.16843)

**Summary**: The paper addresses the trustworthiness of retrieval-augmented large language models, which often suffer from hallucinations due to conflicts between contextual and parametric knowledge. The authors propose a reinforcement learning-based algorithm, Trustworthy-Alignment, to align these models to rely solely on external evidence, thereby enhancing their trustworthiness. The approach demonstrates the models' ability to achieve this alignment without explicit supervision, broadening the scope of alignment techniques from human preference to creating reliable AI agents.

### 22. [Exploring Forgetting in Large Language Model Pre-Training](https://arxiv.org/pdf/2410.17018)

**Summary**: The paper investigates catastrophic forgetting during the pre-training phase of LLMs, challenging traditional metrics like perplexity and proposing new methods to measure entity memory retention. It also explores low-cost strategies to mitigate forgetting and provides insights into the dynamics of forgetting through detailed analysis of learning curves, aiming to advance future LLM research.

### 23. [Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes](https://arxiv.org/pdf/2410.16930)

**Summary**: The paper introduces Math Neurosurgery (MathNeuro), a method for isolating math-specific parameters in Large Language Models (LLMs) using only forward passes. By identifying and scaling these parameters, MathNeuro enhances math reasoning performance by 4-17% on GSM8K tasks without affecting general language abilities. The method is data-efficient, demonstrating effectiveness even with a single sample for parameter identification.

### 24. [Large Language Models Empowered Personalized Web Agents](https://arxiv.org/pdf/2410.17236)

**Summary**: The paper introduces a new approach to enhancing Web agents by integrating personalized data with user instructions, enabling more accurate and customized task execution. It introduces the Personalized Web Agent Benchmark (PersonalWAB) for evaluating these agents and proposes the PUMA framework, which uses a memory bank to align LLMs with personalized user data, resulting in superior performance compared to existing methods.

### 25. [Improving Causal Reasoning in Large Language Models: A Survey](https://arxiv.org/pdf/2410.16676)

**Summary**: The paper surveys methods to improve causal reasoning (CR) in LLMs, categorizing approaches based on whether LLMs act as reasoning engines or provide support to traditional CR methods. It evaluates LLMs' performance on CR tasks, offering insights and suggesting future research directions to enhance their causal reasoning capabilities.

### 26. [UnStar: Unlearning with Self-Taught Anti-Sample Reasoning for LLMs](https://arxiv.org/pdf/2410.17050)

**Summary**: The paper introduces UnSTAR, a method for unlearning in LLMs using self-taught anti-sample reasoning. It proposes the novel concept of anti-samples to reverse learned associations and enable fine-grained, targeted unlearning. The approach demonstrates efficient and selective removal of specific knowledge without affecting related information, offering a new direction for privacy-preserving machine learning and model modification.

### 27. [Towards Reliable Evaluation of Behavior Steering Interventions in LLMs](https://arxiv.org/pdf/2410.17245)

**Summary**: The paper proposes a more reliable evaluation framework for assessing the effectiveness of behavior steering interventions in LLMs. It emphasizes the need for using task-relevant contexts, considering model likelihoods, enabling standardized comparisons, and providing baseline comparisons. The authors introduce an evaluation pipeline based on these criteria and apply it to evaluate two representation engineering methods, revealing that some interventions are less effective than previously reported.

### 28. [Altogether: Image Captioning via Re-aligning Alt-text](https://arxiv.org/pdf/2410.17251)

**Summary**: The paper introduces Altogether, a method for improving image captioning by re-aligning existing alt-text metadata with image content through iterative human annotation. This approach generates richer captions by leveraging pre-existing textual information, leading to enhanced performance in text-to-image generation and zero-shot image classification tasks.

### 29. [Levels of AI Agents: from Rules to Large Language Models](https://arxiv.org/pdf/2405.06643)

**Summary**: The paper categorizes AI agents into six levels, mirroring the levels of autonomous driving, ranging from no AI (L0) to rule-based AI (L1), IL/RL-based AI with reasoning (L2), LLM-based AI with memory (L3), autonomous learning (L4), and multi-agent collaboration with personality (L5). This framework outlines the progression from simple rule-based systems to advanced, self-learning agents capable of complex interactions and emotional intelligence.

### 30. [Merging LoRAs like Playing LEGO: Pushing the Modularity of LoRA to Extremes Through Rank-Wise Clustering](https://arxiv.org/pdf/2409.16167)

**Summary**: The paper introduces the LoRA-LEGO framework, which leverages the concept of Minimal Semantic Units (MSUs) to merge multiple Low-Rank Adaptation (LoRA) models by clustering their rank-wise parameters, akin to assembling LEGO blocks. This approach allows for flexible and efficient combination of LoRAs, outperforming existing methods in enhancing large language models' capabilities across various benchmarks.

### 31. [Context-Parametric Inversion: Why Instruction Finetuning May Not Actually Improve Context Reliance](https://arxiv.org/pdf/2410.10796)

**Summary**: The paper identifies a phenomenon called **context-parametric inversion**, where instruction finetuning initially improves a model's reliance on input context but then causes it to gradually decrease, despite continued performance gains on benchmarks. This inversion occurs because finetuning examples often align with the model's pre-existing knowledge, reducing its need to rely on new context. The study suggests potential mitigation strategies but highlights the need for further research to address this issue in instruction finetuning.

### 32. [Exploring Prompt Engineering: A Systematic Review with SWOT Analysis](https://arxiv.org/pdf/2410.12843)

**Summary**: The paper presents a systematic review and SWOT analysis of prompt engineering techniques for Large Language Models, focusing on linguistic principles to enhance AI interactions. It identifies strengths, weaknesses, opportunities, and threats of various techniques, such as template-based approaches and fine-tuning, and suggests future research directions to improve human-machine communication.

### 33. [TextLap: Customizing Language Models for Text-to-Layout Planning](https://arxiv.org/pdf/2410.12844)

**Summary**: The paper introduces TextLap, a method that customizes LLMs to generate graphical layouts from text instructions. By leveraging a specialized dataset called InsLap, TextLap effectively outperforms existing models, including GPT-4, in tasks related to image generation and graphical design.

### 34. [A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions](https://arxiv.org/pdf/2410.12837)

**Summary**: The paper provides a thorough overview of Retrieval-Augmented Generation (RAG), detailing its evolution, current advancements, and future directions. It highlights how RAG integrates retrieval mechanisms with generative models to enhance accuracy in knowledge-intensive tasks, while also addressing challenges like scalability and ethical concerns. The survey aims to guide researchers and practitioners in leveraging RAG's potential in natural language processing.

### 35. [Navigating the Cultural Kaleidoscope: A Hitchhiker's Guide to Sensitivity in Large Language Models](https://arxiv.org/pdf/2410.12880)

**Summary**: The paper addresses the critical issue of cultural sensitivity in LLMs, particularly in smaller models that may lack comprehensive cultural training data. It introduces a cultural harm test dataset and a culturally aligned preference dataset to evaluate and fine-tune LLMs, ensuring they respect diverse cultural norms. The study demonstrates that incorporating culturally aligned feedback significantly improves model behavior, making LLMs more inclusive and ethically sound in their global applications.

### 36. [Merge to Learn: Efficiently Adding Skills to Language Models with Model Merging](https://arxiv.org/pdf/2410.12937)

**Summary**: The paper explores a cost-effective method for adding new skills to language models by training on new skills separately and then merging them with the existing model, using techniques like task vectors. This "parallel-train-then-merge" approach is shown to be comparable in effectiveness to retraining the model on updated data mixtures, particularly in enhancing safety features without compromising the model's ability to refuse harmful prompts.

### 37. [Self-Pluralising Culture Alignment for Large Language Models](https://arxiv.org/pdf/2410.12971)

**Summary**: The paper introduces CultureSPA, a framework designed to align LLMs with pluralistic human values across diverse cultures. By generating culture-related questions and comparing model outputs in culture-aware and culture-unaware settings, CultureSPA identifies culture-specific instances for fine-tuning, enhancing the model's alignment with various cultures. Experiments show that CultureSPA improves cultural alignment without compromising general model performance, with further enhancements possible through advanced prompt engineering.

### 38. [LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks](https://arxiv.org/pdf/2410.13025)

**Summary**: The paper introduces "LoRA Soups," a method for merging Low-Rank Adaptation (LoRA) modules to compose skills in Large Language Models (LLMs) for practical tasks. The study demonstrates that concatenating LoRAs (CAT) outperforms existing merging techniques, achieving significant improvements in tasks like solving math-word problems and creating question-answering bots. This work highlights model merging as an efficient approach for skill composition, particularly when training data is scarce.

### 39. [The Geometry of Numerical Reasoning: Language Models Compare Numeric Properties in Linear Subspaces](https://arxiv.org/pdf/2410.13194)

**Summary**: The paper explores how LLMs use numerical attributes encoded in low-dimensional subspaces of the embedding space for logical comparison tasks. By identifying these subspaces through partial least squares regression and manipulating hidden states within them, the study demonstrates that LLMs rely on linearly encoded numerical information to make comparison decisions across various attributes.

### 40. [MCQG-SRefine: Multiple Choice Question Generation and Evaluation with Iterative Self-Critique, Correction, and Comparison Feedback](https://arxiv.org/pdf/2410.13191)

**Summary**: The paper introduces MCQG-SRefine, a framework that uses LLMs to generate high-quality multiple-choice questions (MCQG) for professional exams like the USMLE. By incorporating expert-driven prompt engineering and iterative self-critique and correction, the framework improves question quality and difficulty, and it also introduces an automatic evaluation metric to replace expert assessments, making the process more efficient and reliable.

### 41. [Breaking Chains: Unraveling the Links in Multi-Hop Knowledge Unlearning](https://arxiv.org/pdf/2410.13274)

**Summary**: The paper identifies a critical vulnerability in current unlearning techniques for LLMs, where multi-hop queries can still retain indirect references to erased knowledge. To address this, the authors propose MUNCH, an uncertainty-based approach that breaks down multi-hop queries into subquestions and uses model uncertainty to enhance unlearning effectiveness. Empirical results show that MUNCH significantly improves the removal of indirect knowledge and can be integrated with existing unlearning methods.

### 42. [Enhancing Fact Retrieval in PLMs through Truthfulness](https://arxiv.org/pdf/2410.13562)

**Summary**: The paper explores enhancing fact retrieval in Pre-trained Language Models (PLMs) by using a helper model to assess the truthfulness of inputs based on PLM hidden states. The approach improves fact retrieval by up to 33%, demonstrating the potential of leveraging hidden state representations to enhance factual knowledge extraction from PLMs.

### 43. [RAG-DDR: Optimizing Retrieval-Augmented Generation Using Differentiable Data Rewards](https://arxiv.org/pdf/2410.13509)

**Summary**: The paper introduces Differentiable Data Rewards (DDR), a method for optimizing Retrieval-Augmented Generation (RAG) systems by aligning data preferences among different RAG modules. DDR uses a rollout approach to evaluate and optimize agents within the RAG system, leading to significant improvements in performance, especially for smaller-scale LLMs that rely heavily on retrieved knowledge. The method also enhances the generation module's ability to extract key information and resolve conflicts between internal and external knowledge.

### 44. [GeoCoder: Solving Geometry Problems by Generating Modular Code through Vision-Language Models](https://arxiv.org/pdf/2410.13510)

**Summary**: The paper introduces GeoCoder, a system that enhances vision-language models' ability to solve geometry problems by generating modular code using a predefined function library. This approach ensures accurate and deterministic calculations, overcoming the limitations of traditional VLMs in handling mathematical operations and formula application. The multimodal retrieval-augmented variant, RAG-GeoCoder, further improves performance by incorporating a non-parametric memory module for function retrieval, resulting in a 16% average improvement on the GeomVerse dataset.

### 45. [LLM-Human Pipeline for Cultural Context Grounding of Conversations](https://arxiv.org/pdf/2410.13727)

**Summary**: The paper introduces a "Cultural Context Schema" to enhance NLP models' understanding of cultural norms in conversations, particularly focusing on Chinese culture. By generating and refining over 110,000 social norm descriptions using LLMs and human verification, the authors create a dataset that significantly improves the performance of downstream tasks like emotion and dialogue act detection.

### 46. [Learning Representations for Reasoning: Generalizing Across Diverse Structures](https://arxiv.org/pdf/2410.13018)

**Summary**: The paper explores methods for improving reasoning models by enabling generalization across diverse knowledge and query structures. It introduces techniques for handling unseen entities and relations in knowledge graphs and proposes solutions for multi-step queries on both knowledge graphs and text. Additionally, the paper presents systems to facilitate machine learning development on structured data, addressing memory bottlenecks and scaling issues.

### 47. [Rethinking Misalignment in Vision-Language Model Adaptation from a Causal Perspective](https://arxiv.org/pdf/2410.12816)

**Summary**: The paper addresses the misalignment issues in Vision-Language models like CLIP during adaptation to specific tasks, identifying both task and data misalignment. It introduces a structural causal model to analyze these issues and proposes Causality-Guided Semantic Decoupling and Classification (CDC) to mitigate the interference of task-irrelevant knowledge, showing consistent effectiveness across various settings.

### 48. [Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization](https://arxiv.org/pdf/2410.12949)

**Summary**: The paper explores how mechanistic interpretability can enhance the precision and robustness of knowledge editing and unlearning in large language models. By focusing on components associated with the lookup-table mechanism for factual recall, the authors demonstrate more robust unlearning and editing that resist relearning and minimize side effects, outperforming baseline methods across different datasets and models.

### 49. [MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models](https://arxiv.org/pdf/2410.13085)

**Summary**: The paper introduces MMed-RAG, a versatile multimodal retrieval-augmented generation system designed to enhance the factuality of Medical Large Vision-Language Models (Med-LVLMs) by addressing issues like factual hallucination and misalignment. The proposed system includes a domain-aware retrieval mechanism, an adaptive context selection method, and a preference fine-tuning strategy, resulting in a 43.8% average improvement in factual accuracy across various medical datasets.

### 50. [Remember, Retrieve and Generate: Understanding Infinite Visual Concepts as Your Personalized Assistant](https://arxiv.org/pdf/2410.13360)

**Summary**: The paper introduces the Retrieval Augmented Personalization (RAP) framework, which enhances multimodal large language models (MLLMs) to function as personalized assistants by integrating user-specific knowledge. The framework involves three steps: storing user-related information in a key-value database (Remember), retrieving relevant data during conversations (Retrieve), and generating personalized responses using the retrieved information (Generate). This approach allows for real-time concept updates and improves the model's ability to handle diverse tasks like personalized image captioning and question answering.

### 51. [A Unified View of Delta Parameter Editing in Post-Trained Large-Scale Models](https://arxiv.org/pdf/2410.13841)

**Summary**: The paper introduces a unified framework for understanding delta parameter editing in post-trained large-scale models, using Riemann sum approximation of the loss function to categorize methods into competitive, decreased, and improved performance classes. It extends existing techniques like DARE and BitDelta, addressing their limitations and enhancing their applicability, supported by experiments on various models.

### 52. [BLT: Can Large Language Models Handle Basic Legal Text?](https://arxiv.org/pdf/2311.09693)

**Summary**: The paper reveals that current LLMs like GPT-4 and Claude struggle with basic legal text tasks, prompting the development of a benchmark to assess their zero-shot performance. Despite initial poor results, fine-tuning on the benchmark significantly improves even small models' accuracy, suggesting potential for enhancing LLMs' reliability in legal contexts.

### 53. [CoLLEGe: Concept Embedding Generation for Large Language Models](https://arxiv.org/pdf/2403.15362)

**Summary**: The paper introduces CoLLEGe, a meta-learning framework designed to enable large language models to quickly learn new concepts from a few examples by generating flexible embeddings. CoLLEGe focuses on next word prediction, making it compatible with language model pretraining, and demonstrates successful concept learning across various real-world tasks without task-specific training.

### 54. [ActiveRAG: Autonomously Knowledge Assimilation and Accommodation through Retrieval-Augmented Agents](https://arxiv.org/pdf/2402.13547)

**Summary**: The paper introduces ActiveRAG, a multi-agent framework that enhances Retrieval-Augmented Generation (RAG) by enabling Large Language Models (LLMs) to actively engage with and learn from retrieved external knowledge. ActiveRAG includes a knowledge assimilation agent for forming understanding and a thought accommodation agent for refining responses, leading to a 10% improvement in question-answering benchmarks and better handling of noisy retrievals.

### 55. [Avoiding Copyright Infringement via Large Language Model Unlearning](https://arxiv.org/pdf/2406.10952)

**Summary**: The paper introduces Stable Sequential Unlearning (SSU), a novel framework for removing copyrighted content from Large Language Models (LLMs) as new requests for content removal arise. SSU identifies and removes specific weight updates related to copyrighted material while maintaining the model's general language capabilities, demonstrating superior performance compared to existing methods.

### 56. [A Systematic Analysis of Large Language Models as Soft Reasoners: The Case of Syllogistic Inferences](https://arxiv.org/pdf/2406.11341)

**Summary**: The paper investigates how Large Language Models (LLMs) perform on syllogistic reasoning tasks, focusing on the impact of chain-of-thought reasoning, in-context learning, and supervised fine-tuning. The study reveals that while in-context learning and fine-tuning enhance accuracy, only fine-tuning effectively reduces reasoning biases without compromising consistency, aligning with cognitive science heuristics.

### 57. [MedCare: Advancing Medical LLMs through Decoupling Clinical Alignment and Knowledge Aggregation](https://arxiv.org/pdf/2406.17484)

**Summary**: The paper introduces MedCare, a novel approach to advancing medical LLMs by decoupling clinical alignment and knowledge aggregation. Through a two-stage progressive fine-tuning pipeline, MedCare effectively integrates diverse medical knowledge while optimizing for alignment tasks, achieving state-of-the-art performance across multiple medical tasks and demonstrating significant improvements over existing models.

### 58. [Can Large Language Models Generate High-quality Patent Claims?](https://arxiv.org/pdf/2406.19465)

**Summary**: The paper investigates the performance of LLMs in generating patent claims, finding that while GPT-4 excels in producing high-quality first independent claims, its performance declines for dependent claims. Fine-tuning improves feature completeness and clarity, but significant human revision is still required for legal robustness.

### 59. [Are Large Language Models Good Classifiers? A Study on Edit Intent Classification in Scientific Document Revisions](https://arxiv.org/pdf/2410.02028)

**Summary**: The paper investigates the effectiveness of LLMs in classification tasks, focusing on edit intent classification (EIC) in scientific document revisions. Through extensive experiments and comparisons, the study provides insights into the performance of LLMs for EIC and other classification tasks, and introduces Re3-Sci2.0, a new dataset for empirical analysis of editing behavior.

### 60. [Efficient In-Domain Question Answering for Resource-Constrained Environments](https://arxiv.org/pdf/2409.17648)

**Summary**: The paper introduces CRAFT (Compute-Efficient RAFT), a method that combines Retrieval Augmented Fine Tuning (RAFT) with Low-Rank Adaptation (LoRA) to enhance the efficiency of in-domain question answering in resource-constrained environments. By reducing fine-tuning and storage requirements, CRAFT achieves faster inference times while maintaining performance comparable to traditional RAG setups, making it suitable for knowledge-intensive QA tasks with limited hardware and internet access.

### 61. [Temporally Consistent Factuality Probing for Large Language Models](https://arxiv.org/pdf/2409.14065)

**Summary**: The paper introduces TeCFaP, a new task to evaluate the temporal consistency of factuality in Large Language Models (LLMs), addressing limitations in existing benchmarks. It proposes TEMP-COFAC, a dataset for this task, and extends existing metrics to measure temporal consistency. The study finds that LLMs generally perform poorly on TeCFaP and introduces CoTSeLF, a framework combining multi-task instruction tuning and consistent-time-sensitive reinforcement learning, which significantly improves performance on this task.

### 62. [Pyramid-Driven Alignment: Pyramid Principle Guided Integration of Large Language Models and Knowledge Graphs](https://arxiv.org/pdf/2410.12298)

**Summary**: The paper introduces Pyramid-Driven Alignment (PDA), a framework that integrates Large Language Models (LLMs) with Knowledge Graphs (KGs) to enhance reasoning and reduce hallucinations. PDA employs a hierarchical pyramid structure guided by the Pyramid Principle to better align LLM and KG knowledge, and uses a recursive mechanism to leverage KG reasoning for more accurate question-answering. Experimental results show significant improvements over existing methods, with gains of up to 26.78%.

### 63. [LLM Processes: Numerical Predictive Distributions Conditioned on Natural Language](https://arxiv.org/pdf/2405.12856)

**Summary**: The paper introduces LLM Processes, a method for generating numerical predictive distributions from Large Language Models (LLMs) conditioned on natural language descriptions of prior knowledge. By leveraging LLMs' ability to understand and incorporate expert insights, the authors demonstrate improved predictive performance in various settings, bridging the gap between qualitative descriptions and quantitative predictions.

### 64. [RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models](https://arxiv.org/pdf/2407.05131)

**Summary**: The paper introduces RULE, a method to enhance the factual accuracy of Medical Large Vision Language Models (Med-LVLMs) by addressing challenges in Retrieval-Augmented Generation (RAG). RULE employs a calibrated selection of retrieved contexts to control factuality risk and fine-tunes the model using a preference dataset to balance reliance on inherent knowledge and retrieved contexts. The approach shows a 47.4% average improvement in factual accuracy across medical VQA and report generation tasks.

### 65. [LLM-based Cognitive Models of Students with Misconceptions](https://arxiv.org/pdf/2410.12294)

**Summary**: The paper explores the potential of Large Language Models (LLMs) to simulate student cognition in algebra by accurately replicating misconceptions while still solving problems correctly. Through the development of MalAlgoPy, a Python library that generates authentic student solution patterns, the study finds that LLMs can be instruction-tuned to meet these dual requirements by carefully balancing the ratio of correct to misconception examples in training data, thereby advancing the development of effective AI-driven educational technologies.

### 66. [Understanding the Interplay between Parametric and Contextual Knowledge for Large Language Models](https://arxiv.org/pdf/2410.08414)

**Summary**: The paper explores how LLMs integrate parametric knowledge (PK) with contextual knowledge (CK) and identifies four types of interactions between them. Through the ECHOQA benchmark, it finds that LLMs often suppress PK in favor of CK, even when PK is more relevant, highlighting a vulnerability in their performance on knowledge-intensive tasks.

### 67. [Parameter-Efficient Fine-Tuning of Large Language Models using Semantic Knowledge Tuning](https://arxiv.org/pdf/2410.08598)

**Summary**: The paper introduces Semantic Knowledge Tuning (SK-Tuning), a novel method for fine-tuning Large Language Models (LLMs) using meaningful words instead of random tokens, which enhances performance on tasks like text classification and understanding. SK-Tuning leverages the LLM's zero-shot capabilities to process semantic content, resulting in faster training times, fewer parameters, and superior performance compared to traditional tuning methods.

### 68. [StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Structurization](https://arxiv.org/pdf/2410.08815)

**Summary**: The paper introduces StructRAG, a novel framework that enhances LLMs for knowledge-intensive reasoning tasks by converting raw information into structured formats at inference time. This approach allows for more accurate identification of key information and improved global reasoning, leading to state-of-the-art performance across various challenging tasks.

### 69. [Retriever-and-Memory: Towards Adaptive Note-Enhanced Retrieval-Augmented Generation](https://arxiv.org/pdf/2410.08821)

**Summary**: The paper introduces Adaptive Note-Enhanced Retrieval-Augmented Generation (Adaptive-Note), a novel approach for complex question-answering tasks that addresses the limitations of existing RAG methods by iteratively collecting and integrating new information into an adaptive memory structure. This approach enhances knowledge interactions and employs a note-based strategy to determine when to stop retrieving information, leading to improved answer quality across multiple datasets.

### 70. [NoVo: Norm Voting off Hallucinations with Attention Heads in Large Language Models](https://arxiv.org/pdf/2410.08970)

**Summary**: The paper introduces Norm Voting (NoVo), a lightweight method that leverages attention head norms in Large Language Models to significantly improve factual accuracy in zero-shot multiple-choice questions. NoVo outperforms current state-of-the-art methods by a substantial margin and demonstrates exceptional generalization across diverse datasets, offering new possibilities for enhancing LLM interpretability and robustness.

### 71. [Consecutive Batch Model Editing with HooK Layers](https://arxiv.org/pdf/2403.05330)

**Summary**: The paper introduces CoachHooK, a model editing method designed to efficiently handle both sequential and batch editing scenarios without requiring excessive memory. By utilizing hook layers that maintain a constant size, CoachHooK addresses the limitations of existing methods, demonstrating superior performance in both single-round and consecutive batch editing scenarios. The method's stability over multiple editing steps is also validated through extensive analysis.

### 72. [Evaluating Copyright Takedown Methods for Language Models](https://arxiv.org/pdf/2406.18664)

**Summary**: The paper introduces CoTaEval, an evaluation framework to assess the effectiveness of copyright takedown methods for language models, focusing on their impact on model utility, efficiency, and retention of factual knowledge. The study finds that no single method excels across all metrics, highlighting the need for further research and unresolved challenges in this area.

### 73. [Modular Pluralism: Pluralistic Alignment via Multi-LLM Collaboration](https://arxiv.org/pdf/2406.15951)

**Summary**: The paper introduces Modular Pluralism, a framework that enhances LLMs by integrating specialized, smaller community LMs to better represent diverse human preferences. This modular approach supports three modes of pluralism and is shown to improve performance across various tasks, particularly in handling value-laden and perspective-informed responses. The framework is compatible with both black-box and open-source LLMs, allowing for the addition of new community LMs to address underrepresentation.

### 74. [Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning](https://arxiv.org/pdf/2404.05868)

**Summary**: The paper introduces Negative Preference Optimization (NPO), a novel method for unlearning undesirable data in Large Language Models (LLMs), which addresses the issues of catastrophic collapse and ineffective unlearning seen in gradient ascent-based approaches. NPO demonstrates superior performance in both synthetic and benchmark datasets, achieving significant unlearning while preserving model utility, and producing more coherent outputs compared to existing methods.

### 75. [OAEI-LLM: A Benchmark Dataset for Understanding Large Language Model Hallucinations in Ontology Matching](https://arxiv.org/pdf/2409.14038)

**Summary**: The paper introduces OAEI-LLM, a benchmark dataset designed to assess and understand hallucinations in LLMs when applied to ontology matching (OM) tasks. It extends existing OAEI datasets to include LLM-specific hallucinations, providing a framework for evaluating and mitigating these issues in OM.

### 76. [Unraveling Cross-Modality Knowledge Conflicts in Large Vision-Language Models](https://arxiv.org/pdf/2410.03659)

**Summary**: The paper addresses the issue of cross-modality parametric knowledge conflicts in Large Vision-Language Models (LVLMs), where inconsistencies between visual and textual components lead to errors. The authors introduce a systematic approach to detect and mitigate these conflicts, including a dynamic contrastive decoding method that improves model accuracy by 2.24% on average using LLaVA-34B.

### 77. [MKGL: Mastery of a Three-Word Language](https://arxiv.org/pdf/2410.07526)

**Summary**: The paper introduces MKGL, a specialized Knowledge Graph Language (KGL) designed to integrate LLMs with knowledge graphs (KGs). By structuring sentences as entity-relation-entity triplets, MKGL minimizes hallucinations and enhances accuracy. The study demonstrates that LLMs can achieve fluency in KGL through tailored learning methods, significantly outperforming traditional KG embedding techniques in KG completion tasks.

### 78. [KRAG Framework for Enhancing LLMs in the Legal Domain](https://arxiv.org/pdf/2410.07551)

**Summary**: The paper introduces the Knowledge Representation Augmented Generation (KRAG) framework, which enhances Large Language Models (LLMs) by incorporating critical knowledge entities and relationships often missing in standard datasets. Specifically, in the legal domain, KRAG is implemented through Soft PROLEG, which uses inference graphs to improve LLMs' ability to provide structured legal reasoning and explanations. This approach significantly advances natural language understanding in specialized domains like law.

### 79. [Uncovering Overfitting in Large Language Model Editing](https://arxiv.org/pdf/2410.07819)

**Summary**: The paper identifies a phenomenon called Editing Overfit in Large Language Models, where edited models overly prioritize the edit target, leading to poor generalization in complex tasks. To address this, the authors introduce a new benchmark, EVOKE, and propose a strategy called Learn to Inference (LTI) with a Multi-stage Inference Constraint module, which helps edited models recall knowledge more effectively, reducing overfitting.

### 80. [Fine-Tuning Language Models for Ethical Ambiguity: A Comparative Study of Alignment with Human Responses](https://arxiv.org/pdf/2410.07826)

**Summary**: The study investigates the alignment of language models with human judgments in morally ambiguous scenarios by fine-tuning models on curated datasets from the Scruples project. Significant improvements in model performance were observed post-fine-tuning, particularly in cross-entropy and Dirichlet scores, highlighting the potential for enhancing ethical reasoning in language models. However, the fine-tuned models still lagged behind BERT and RoBERTa in cross-entropy scores, suggesting ongoing challenges in capturing human judgment nuances.

### 81. [Disease Entity Recognition and Normalization is Improved with Large Language Model Derived Synthetic Normalized Mentions](https://arxiv.org/pdf/2410.07951)

**Summary**: The study demonstrates that fine-tuning a Large Language Model (LLM) to generate synthetic normalized mentions of disease entities significantly improves Disease Entity Normalization (DEN) performance, particularly for out-of-distribution data, while yielding only modest enhancements for Disease Entity Recognition (DER). The synthetic data augmentation led to substantial accuracy improvements in DEN across multiple datasets, suggesting that LLM-generated data can effectively supplement traditional training methods.

### 82. [A Closer Look at Machine Unlearning for Large Language Models](https://arxiv.org/pdf/2410.08109)

**Summary**: The paper examines the challenges of machine unlearning in LLMs, particularly in removing specific content while maintaining model performance. It introduces new evaluation metrics and proposes methods like maximizing entropy for untargeted unlearning and answer preservation loss for targeted unlearning to address these challenges. Experimental results show the effectiveness of these approaches across various unlearning scenarios.

### 83. [Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study over Open-ended Question Answering](https://arxiv.org/pdf/2410.08085)

**Summary**: The paper introduces OKGQA, a new benchmark for evaluating the effectiveness of Knowledge Graphs (KGs) in enhancing Large Language Models (LLMs) in open-ended question answering scenarios. The benchmark aims to assess both the reduction in hallucinations and the improvement in reasoning capabilities of LLMs when integrated with KGs, and it also includes a perturbed version (OKGQA-P) to test model robustness against KG errors. The study seeks to determine whether KGs can make LLMs more trustworthy in real-world applications and to guide future research in this area.

### 84. [Insight Over Sight? Exploring the Vision-Knowledge Conflicts in Multimodal LLMs](https://arxiv.org/pdf/2410.08145)

**Summary**: The paper investigates the issue of vision-knowledge conflicts in Multimodal Large Language Models (MLLMs), where visual information contradicts the model's internal commonsense knowledge. It introduces a benchmark with 374 images and 1,122 QA pairs to assess conflict resolution, finding that models often over-rely on textual queries. A new prompting strategy, "Focus-on-Vision" (FoV), is proposed to improve models' reliance on visual data, enhancing their conflict-resolution capabilities.

### 85. [From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions](https://arxiv.org/pdf/2410.08197)

**Summary**: The paper introduces DRAFT, a framework that dynamically refines tool documentation for Large Language Models (LLMs) by analyzing feedback and interaction trails. Through a trial-and-error approach involving experience gathering, learning, and documentation rewriting, DRAFT iteratively improves documentation quality, enhancing LLMs' understanding and use of tools. The framework's effectiveness is demonstrated through experiments showing improved documentation quality and cross-model generalization.

### 86. [Mitigating Gender Bias in Code Large Language Models via Model Editing](https://arxiv.org/pdf/2410.07820)

**Summary**: The paper introduces CodeGenBias, a dataset and FB-Score metric to evaluate gender bias in code Large Language Models (LLMs), and proposes MG-Editing, a multi-granularity model editing approach to mitigate this bias. Experiments show that MG-Editing effectively reduces gender bias while preserving code generation capabilities, with the best performance at row and neuron levels of granularity.

### 87. [Composite Learning Units: Generalized Learning Beyond Parameter Updates to Transform LLMs into Adaptive Reasoners](https://arxiv.org/pdf/2410.08037)

**Summary**: The paper introduces Composite Learning Units (CLUs) to enhance Large Language Models (LLMs) with continuous, generalized learning capabilities without traditional parameter updates. CLUs utilize a dynamic knowledge repository, including a General Knowledge Space and a Prompt-Specific Knowledge Space, to iteratively refine reasoning through goal-driven interactions and feedback, enabling adaptive reasoning and autonomous learning from past experiences.

### 88. [SLIM: Let LLM Learn More and Forget Less with Soft LoRA and Identity Mixture](https://arxiv.org/pdf/2410.07739)

**Summary**: The paper introduces SLIM, a novel mixture of expert framework that combines Soft LoRA and Identity Mixture to efficiently fine-tune LLMs while mitigating catastrophic forgetting. SLIM dynamically routes between LoRA adapters and skipping connections, enhancing out-of-domain distinction and preserving the base model's general capabilities. Experimental results show that SLIM outperforms existing parameter-efficient fine-tuning methods in balancing downstream task performance and preventing forgetting.

### 89. [Fuse to Forget: Bias Reduction and Selective Memorization through Model Fusion](https://arxiv.org/pdf/2311.07682)

**Summary**: The paper explores the use of model fusion to reduce unwanted knowledge, such as shortcuts, social biases, and memorization of training data, in fine-tuned language models. Through experiments, it shows that model fusion enhances shared knowledge while forgetting unshared knowledge, making it a promising tool for debiasing and addressing privacy concerns in language models.

### 90. [AKEW: Assessing Knowledge Editing in the Wild](https://arxiv.org/pdf/2402.18909)

**Summary**: The paper introduces AKEW, a benchmark for assessing knowledge editing in practical scenarios, addressing the limitations of current evaluations that focus on structured facts from curated datasets. AKEW includes settings for structured facts, unstructured texts, and extracted triplets, and features both counterfactual and real-world knowledge updates, revealing significant gaps between existing methods and practical needs.

### 91. [AutoRD: An Automatic and End-to-End System for Rare Disease Knowledge Graph Construction Based on Ontologies-enhanced Large Language Models](https://arxiv.org/pdf/2403.00953)

**Summary**: The paper introduces AutoRD, an end-to-end system designed to automate the construction of a knowledge graph for rare diseases by extracting entities and their relations from medical texts. Leveraging ontologies-enhanced Large Language Models, AutoRD integrates up-to-date structured knowledge and outperforms traditional methods and common LLMs in rare disease extraction tasks.

### 92. [LLaMP: Large Language Model Made Powerful for High-fidelity Materials Knowledge Retrieval and Distillation](https://arxiv.org/pdf/2401.17244)

**Summary**: The paper introduces LLaMP, a multimodal retrieval-augmented generation framework that enhances Large Language Models (LLMs) for high-fidelity materials knowledge retrieval and distillation. By integrating hierarchical reasoning-and-acting agents with computational and experimental data, LLaMP effectively reduces hallucination and biases in LLMs, demonstrating strong tool usage and self-consistency in handling complex materials science tasks. The framework also showcases capabilities in editing crystal structures and running molecular dynamics simulations, offering a nearly hallucination-free approach to materials informatics.

### 93. [Unlocking the Power of Large Language Models for Entity Alignment](https://arxiv.org/pdf/2402.15048)

**Summary**: The paper introduces ChatEA, a novel framework that leverages LLMs to enhance Entity Alignment (EA) in knowledge graphs. By translating KG structures into a format understandable by LLMs and employing a two-stage EA strategy, ChatEA improves EA accuracy by utilizing LLMs' extensive background knowledge and multi-step reasoning capabilities. Experimental results demonstrate ChatEA's superior performance over traditional methods.

### 94. [Verification and Refinement of Natural Language Explanations through LLM-Symbolic Theorem Proving](https://arxiv.org/pdf/2405.01379)

**Summary**: The paper introduces Explanation-Refiner, a neuro-symbolic framework that integrates Large Language Models (LLMs) with Theorem Provers (TPs) to verify and refine natural language explanations for Natural Language Inference (NLI). By leveraging TPs for formal validation and feedback, the framework aims to enhance the quality and logical correctness of explanations across various domains, thereby addressing the limitations of current crowd-sourced validation methods.

### 95. [ClaimBrush: A Novel Framework for Automated Patent Claim Refinement Based on Large Language Models](https://arxiv.org/pdf/2410.05575)

**Summary**: The paper introduces ClaimBrush, a framework for automated patent claim refinement using large language models. It includes a dataset of actual patent claim rewriting cases and a fine-tuned rewriting model, enhanced by preference optimization based on patent examiners' feedback. Experimental results demonstrate superior performance over heuristic baselines and zero-shot learning methods.

### 96. [Dr-LLaVA: Visual Instruction Tuning with Symbolic Clinical Grounding](https://arxiv.org/pdf/2405.19567)

**Summary**: The paper introduces Dr-LLaVA, a Vision-Language Model (VLM) designed for medical applications, which uses a novel alignment algorithm to ground its outputs in clinical reasoning. This algorithm leverages symbolic representations of medical knowledge to generate training data and create an automatic reward function, eliminating the need for human involvement and reducing costs. Dr-LLaVA demonstrates strong performance in multi-turn medical conversations, particularly in analyzing bone marrow pathology slides.

### 97. [Can Large Language Models Understand DL-Lite Ontologies? An Empirical Study](https://arxiv.org/pdf/2406.17532)

**Summary**: The paper investigates whether LLMs can understand DL-Lite ontologies, focusing on six representative tasks from syntactic and semantic perspectives. The study reveals that LLMs effectively grasp formal syntax and model-theoretic semantics but face challenges with TBox NI transitivity and large ABoxes. The findings aim to inform future knowledge engineering solutions.

### 98. [Revisiting the Superficial Alignment Hypothesis](https://arxiv.org/pdf/2410.03717)

**Summary**: The paper challenges the Superficial Alignment Hypothesis by demonstrating that post-training with increasing fine-tuning examples significantly improves language model performance across various tasks, including mathematical reasoning and multihop reasoning. The study reveals that model performance scales as a power law with the number of fine-tuning examples, indicating that post-training is crucial for enhancing reasoning abilities and integrating new knowledge, contrary to the hypothesis's claims.

### 99. [Neurosymbolic AI approach to Attribution in Large Language Models](https://arxiv.org/pdf/2410.03726)

**Summary**: The paper proposes a Neurosymbolic AI (NesyAI) approach to improve attribution in LLMs, addressing issues like hallucinations, biases, and unreliable sources by combining neural networks with structured symbolic reasoning. This integration aims to provide more reliable, interpretable, and adaptable systems for ensuring the factual accuracy and reliability of LLM outputs.

### 100. [Reasoning Elicitation in Language Models via Counterfactual Feedback](https://arxiv.org/pdf/2410.03767)

**Summary**: The paper introduces new metrics to evaluate the reasoning capabilities of language models, particularly in counterfactual scenarios, and proposes fine-tuning methods to enhance these abilities. The study evaluates the fine-tuned models across various reasoning tasks, demonstrating improved generalization and performance compared to baseline models.



---

*Last updated on 2024-10-26*