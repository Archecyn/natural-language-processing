# natural-language-processing
NLP techniques from foundational text processing to advanced transformer-based models. Covering tokenisation, embeddings, sentiment analysis, named entity recognition, and sequence modelling with Python and HuggingFace.

```
natural-language-processing/
├── README.md
├── requirements.txt
├── 01_fundamentals/
│   ├── text_preprocessing.ipynb        # Tokenisation, stemming, lemmatisation
│   ├── bag_of_words.ipynb              # BoW, N-grams, vocabulary
│   └── tfidf.ipynb                     # TF-IDF, cosine similarity
├── 02_embeddings/
│   ├── word2vec.ipynb                  # Skip-gram, CBOW from scratch
│   ├── glove.ipynb                     # Global vectors, co-occurrence
│   └── sentence_transformers.ipynb     # Semantic similarity, SBERT
├── 03_classification/
│   ├── sentiment_analysis.ipynb        # Lexicon-based + ML approaches
│   ├── topic_modelling.ipynb           # LDA, NMF
│   └── named_entity_recognition.ipynb  # spaCy, HuggingFace NER
├── 04_transformers/
│   ├── attention_mechanism.ipynb       # Self-attention from scratch
│   ├── bert_finetuning.ipynb           # HuggingFace fine-tuning
│   ├── text_summarisation.ipynb        # Extractive & abstractive
│   └── question_answering.ipynb        # Extractive QA, SQuAD
├── 05_llm_engineering/
│   ├── prompt_engineering.ipynb        # Zero-shot, few-shot, CoT
│   ├── aws_bedrock_llm.ipynb           # Bedrock API, model comparison
│   ├── rag_pipeline.ipynb              # Chunking, embeddings, retrieval
│   ├── langgraph_agents.ipynb          # Stateful agentic workflows
│   ├── finetuning.ipynb                # LoRA, QLoRA, PEFT
│   └── databricks_agent_bricks.ipynb   # Agent deployment on Databricks
└── 06_projects/
    ├── news_classifier/                # End-to-end classification pipeline
    ├── semantic_search_engine/         # Embeddings + vector store + UI
    └── agentic_rag_pipeline/           # LangGraph + RAG end-to-end
```