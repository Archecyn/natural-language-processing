# natural-language-processing
NLP techniques from foundational text processing to advanced transformer-based models. Covering tokenisation, embeddings, sentiment analysis, named entity recognition, and sequence modelling with Python and HuggingFace.

natural-language-processing/
├── README.md
├── requirements.txt
├── 01_fundamentals/
│   ├── text_preprocessing.ipynb   # Tokenisation, stemming, lemmatisation
│   ├── bag_of_words.ipynb
│   └── tfidf.ipynb
├── 02_embeddings/
│   ├── word2vec.ipynb
│   ├── glove.ipynb
│   └── sentence_transformers.ipynb
├── 03_classification/
│   ├── sentiment_analysis.ipynb
│   ├── topic_modelling.ipynb      # LDA, NMF
│   └── named_entity_recognition.ipynb
├── 04_transformers/
│   ├── bert_finetuning.ipynb      # HuggingFace
│   ├── text_summarisation.ipynb
│   ├── question_answering.ipynb
│   └── aws_bedrock_llm.ipynb             # 🆕 Calling LLMs via AWS Bedrock
├── 05_projects/
│   ├── news_classifier/
│   ├── semantic_search_engine/
│   ├── prompt_engineering.ipynb
│   ├── rag_pipeline.ipynb                # Retrieval-Augmented Generation
│   ├── langgraph_agents.ipynb            # 🆕 LangGraph agentic workflows
│   ├── finetuning.ipynb
│   └── databricks_agent_bricks.ipynb     # 🆕 Agent deployment on Databricks
└── 06_projects/
    └── agentic_rag_pipeline/             # 🆕 LangGraph + RAG end-to-end