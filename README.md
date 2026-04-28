# 🧠 Job Recommendation System
### Data Science & Machine Learning Internship Project

> An end-to-end content-based job recommendation system that matches candidates to the most relevant job postings using NLP, multi-metric similarity scoring, and an interactive Streamlit dashboard — built in the Sri Lankan recruitment context.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Setup & Installation](#setup--installation)
- [Running the Notebooks](#running-the-notebooks)
- [Running the Dashboard](#running-the-dashboard)
- [Hybrid Scoring Formula](#hybrid-scoring-formula)
- [Evaluation Results](#evaluation-results)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

This project recommends the most suitable job postings to candidates based on their skills, experience, domain, and location preferences. It uses **Content-Based Filtering** with a hybrid scoring model that combines:

- **TF-IDF cosine similarity** — keyword-level skill matching
- **Word2Vec semantic similarity** — captures related skills even when keywords differ
- **Structural bonuses** — experience level, domain/industry match, and location preference

The system was built as part of a 2-week Data Science internship at **Gamage Recruiters**, covering the full ML pipeline from data generation to an interactive deployment.

---

## Project Structure

```
job-recommendation-system/
│
├── data/                                   # Raw and preprocessed datasets
│   ├── candidates.csv                      # 500 synthetic candidate profiles
│   ├── job_postings.csv                    # 200 synthetic job postings
│   ├── candidates_preprocessed.csv         # Cleaned + encoded candidate data
│   └── job_postings_preprocessed.csv       # Cleaned + encoded job data
│
├── models/                                 # Saved models and matrices
│   ├── candidate_tfidf_matrix.npz          # Sparse TF-IDF matrix (500 × 300)
│   ├── job_tfidf_matrix.npz                # Sparse TF-IDF matrix (200 × 300)
│   ├── tfidf_vectorizer.pkl                # Fitted TF-IDF vectorizer
│   ├── word2vec_skills.model               # Trained Word2Vec model
│   ├── candidate_w2v_embeddings.npy        # W2V embeddings (500 × 100)
│   ├── job_w2v_embeddings.npy              # W2V embeddings (200 × 100)
│   ├── step6_hybrid_score_matrix.npy       # Hybrid score matrix (500 × 200)
│   ├── step6_tfidf_cosine_matrix.npy
│   ├── step6_w2v_cosine_matrix.npy
│   ├── step6_euc_sim_matrix.npy
│   └── step6_pearson_sim_matrix.npy
│
├── outputs/                                # Generated recommendation outputs
│   ├── step6_recommendations.csv           # Top-5 jobs per candidate (2,500 rows)
│   └── step7_eval_results.json             # Evaluation metrics for the dashboard
│
├── notebooks/                              # Jupyter notebooks (Steps 2–7)
│   ├── step2_data_generate.ipynb
│   ├── step3_preprocessing.ipynb
│   ├── step4_recommendation_engine.ipynb
│   ├── step5_text_to_features.ipynb
│   ├── step6_similarity_matching.ipynb
│   └── step7_evaluation.ipynb
│
├── step8_dashboard.py                      # Streamlit dashboard (Step 8)
├── requirements.txt
├── Project_Report.pdf
└── README.md
```

---

## Pipeline

```
Step 2              Step 3              Step 4              Step 5
Data Generation ──▶ Preprocessing  ──▶ Content-Based  ──▶ Text Features
500 candidates       Text cleaning       Filtering           TF-IDF analysis
200 job postings     TF-IDF vectors      Score matrix        Word2Vec training
Sri Lankan context   Encoding / norm     Top-N ranking       Embedding comparison
       │
       ▼
Step 6              Step 7              Step 8
Similarity      ──▶ Evaluation      ──▶ Dashboard
Matching            Precision@K         Streamlit + Plotly
4 metrics           Recall@K            Candidate Explorer
Hybrid score        NDCG, MAP           Bulk recommendations
                    Hit Rate            Interactive filters
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt

```
pandas
numpy
scikit-learn
gensim
scipy
matplotlib
seaborn
faker
streamlit
plotly
jupyter
```

---

## Running the Notebooks

Run the notebooks **in order** — each step saves outputs consumed by the next.

```bash
cd notebooks/

# Step 2: Generate synthetic dataset
jupyter notebook step2_data_generate.ipynb

# Step 3: Preprocess candidates and job postings
jupyter notebook step3_preprocessing.ipynb

# Step 4: Build content-based recommendation engine
jupyter notebook step4_recommendation_engine.ipynb

# Step 5: TF-IDF analysis + Word2Vec embeddings
jupyter notebook step5_text_to_features.ipynb

# Step 6: Multi-metric similarity matching + hybrid scoring
jupyter notebook step6_similarity_matching.ipynb

# Step 7: Evaluate recommendations (Precision@K, MAP, NDCG, etc.)
jupyter notebook step7_evaluation.ipynb
```

> ⚠️ Make sure `data/`, `models/`, and `outputs/` directories exist before running. Each notebook creates them automatically if they are missing.

---

## Running the Dashboard

After running all notebooks (Steps 2–7), launch the interactive dashboard:

```bash
streamlit run step8_dashboard.py
```

The dashboard opens at `http://localhost:8501` and includes:

| Tab | Description |
|-----|-------------|
| **Candidate Explorer** | Select any candidate → view profile + ranked job recommendations + similarity score chart |
| **Model Evaluation** | Interactive Precision@K, Recall@K, MAP, NDCG, Hit Rate charts from Step 7 |
| **Bulk Recommendations** | Top-N jobs for up to 100 candidates; wide and long table views; score heatmap |
| **About** | Pipeline summary, hybrid formula, dataset statistics |

---

## Hybrid Scoring Formula

```
Hybrid Score = 0.55 × TF-IDF Cosine
             + 0.20 × Word2Vec Cosine
             + Experience Bonus (0.15 / 0.07 / 0.00)
             + Domain Bonus    (0.10 / 0.00)
             + Location Bonus  (0.05 / 0.00)
```

| Component | Weight / Value | Signal |
|-----------|---------------|--------|
| TF-IDF Cosine | 0.55 | Exact keyword skill matching |
| Word2Vec Cosine | 0.20 | Semantic skill similarity |
| Experience (exact match) | +0.15 | Candidate level == Job level |
| Experience (1 level off) | +0.07 | \|candidate − job level\| == 1 |
| Domain/Industry match | +0.10 | Candidate domain == Job industry |
| Location preference | +0.05 | Job city ∈ candidate's preferred locations |

---

## Evaluation Results

Evaluation was performed on all five similarity metrics using **domain match** as a binary relevance proxy (job industry == candidate primary domain).

| Metric | Precision@5 | Recall@5 | NDCG@5 | MAP |
|--------|-------------|----------|--------|-----|
| **Hybrid Score** | **Best** | **Best** | **Best** | **Best** |
| TF-IDF Cosine | High | Moderate | High | High |
| Word2Vec Cosine | Moderate | Moderate | Moderate | Moderate |
| Euclidean Sim | Moderate | Moderate | Moderate | Moderate |
| Pearson Sim | Lower | Lower | Lower | Lower |

> The Hybrid Score outperformed all individual metrics on MAP and NDCG across all K values (1, 3, 5, 10).

**Key findings:**
- Precision decreases and Recall improves as K increases — expected IR behaviour.
- NDCG confirms the Hybrid Score places relevant jobs higher in the ranked list, not just includes more of them.
- Hit Rate@5 is high across all metrics — most candidates receive at least one domain-relevant recommendation.
- Technology and Finance domains see higher Precision@5 due to larger job pools in those sectors.

---

## Dataset

The dataset is **fully synthetic**, generated using Python's `Faker` library with Sri Lankan contextual data.

| | Candidates | Job Postings |
|-|-----------|-------------|
| **Records** | 500 | 200 |
| **Domains** | Technology, Finance, Marketing, HR, Operations | Same 5 domains |
| **Locations** | Colombo, Kandy, Galle, Negombo, Jaffna, Kurunegala | Same 6 cities |
| **Salary** | — | LKR (Sri Lankan Rupee) |
| **Skills per record** | 4–12 | 3–10 required + preferred |
| **Experience levels** | Entry, Junior, Mid, Senior, Lead, Manager | Same levels |

No real personal data is used. All names, contact details, and company names are synthetic.

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` / `numpy` | Data manipulation and matrix operations |
| `scikit-learn` | TF-IDF vectorisation, cosine similarity, encoders, MinMaxScaler |
| `gensim` | Word2Vec training and skill embeddings |
| `scipy` | Sparse matrix storage (TF-IDF) |
| `Faker` | Synthetic Sri Lankan dataset generation |
| `matplotlib` / `seaborn` | Static visualisations in notebooks |
| `Streamlit` | Interactive dashboard web application |
| `Plotly` | Interactive charts in the dashboard |

---

## Limitations & Future Work

**Current limitations:**
- Synthetic data may not capture the full complexity and noise of real-world resumes.
- Evaluation uses a domain-match proxy — no historical application data was available.
- Word2Vec is trained on a small corpus (700 skill sequences); rare skills may have low-quality embeddings.
- No collaborative filtering component.

**Planned improvements:**
- Replace Word2Vec with pre-trained sentence embeddings (e.g., `sentence-transformers`, BERT).
- Add collaborative filtering using historical application patterns.
- Collect real anonymised candidate and job data for training and validation.
- Build a candidate feedback loop (thumbs up/down on recommendations) for online learning.
- Deploy as a production web app with authentication and database integration.

---

## Author

Nisansala Ruwan Pathirana
Data Science & Machine Learning Division  
April 2026
