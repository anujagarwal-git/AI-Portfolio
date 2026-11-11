# AI Portfolio — Credit Risk × AI

Applied AI and machine learning projects on banking and credit-risk problems, from traditional machine learning to deep learning and a regulatory RAG assistant.

The projects are built from a **credit-risk practitioner's point of view**: the focus is on credit judgement, leakage control, validation and explainability, not only model scores.

---

## Projects

| Area | Project | What it does | Key result |
|---|---|---|---|
| **GenAI / RAG** | [RegRAG — Regulatory Assistant for Credit Risk](RAG/) | Answers questions over 19 credit and model-risk regulations (Basel, IFRS 9, PRA SS1/23, SR 11-7, BCBS 239, US stress testing). Every claim cites its source paragraph. Runs fully on-premise. | 96% retrieval hit rate · RAGAS faithfulness 0.89 · 104/104 citations valid |
| **Deep Learning** | [Credit Default (PD) Model](Deep-Learning/02-credit-default-pd-model/) | Champion–challenger PD model on 1.3M LendingClub loans: XGBoost vs a neural network, using only data available at approval. | AUC 0.70 · KS 28.9% |
| **Deep Learning** | [Credit Card Fraud Detection](Deep-Learning/01-credit-card-fraud-detection/) | Neural network for fraud that is 0.17% of transactions, with leakage-proof cross-validation and threshold tuning. | Precision 0.77 · recall 0.74 · F1 0.76 |
| **Traditional ML** | [Term Deposit Propensity Model](Traditional-ML/02-bank-marketing-propensity/) | Predicts which customers will buy a term deposit, comparing nine feature-selection methods; call duration excluded to avoid leakage. | F1 0.44 · AUC 0.74 |
| **Traditional ML** | [Loan Approval Prediction](Traditional-ML/01-loan-approval-prediction/) | Predicts home-loan approval with logistic regression, decision tree and XGBoost pipelines. | F1 0.90 |

Each project folder has its own README with the approach, results and limitations.

---

## What runs through the projects

- **Credit judgement in the data:** removing look-ahead variables (bureau data pulled after origination, call duration) so models only use what is known at decision time.
- **Honest validation:** leakage-proof cross-validation, champion–challenger comparison, and limitations stated openly (for example, in-time vs out-of-time PSI).
- **Credit-risk metrics:** ROC-AUC / Gini, KS, PSI, and precision–recall trade-offs for imbalanced targets.
- **Explainability and governance:** SHAP risk drivers, source citations and audit logs for GenAI answers.
- **Bank constraints:** the RAG assistant runs entirely on local infrastructure, so no data leaves the bank's systems.

---

## Repository layout

```
AI-Portfolio/
├── RAG/                                   Regulatory RAG assistant
├── Deep-Learning/
│   ├── 01-credit-card-fraud-detection/
│   └── 02-credit-default-pd-model/
└── Traditional-ML/
    ├── 01-loan-approval-prediction/
    └── 02-bank-marketing-propensity/
```

## Tools

- **Machine learning:** Python, pandas, scikit-learn, XGBoost, TensorFlow / Keras, SciKeras, SHAP
- **GenAI / RAG:** Docling, bge embeddings, Qdrant, cross-encoder reranking, Ollama (Qwen), RAGAS, Arize Phoenix
- **Serving:** FastAPI, Streamlit, Docker

## Coming next
- Agentic AI with LangGraph

---

**Author:** Anuj Agarwal, credit risk professional (11 years in retail lending) applying AI to banking problems.
