# Yelp Big Data Intelligence & Hybrid Recommender System 🍕📊

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An end-to-end data science and machine learning research pipeline applied to the **Yelp Academic Dataset** (150,000+ businesses, millions of customer reviews). This project explores multi-stage data ingestion, geospatial GIS density mapping, NLP sentiment analysis, topic modeling, and hybrid collaborative filtering recommendation algorithms.

---

## 🌟 Architecture & Pipeline Workflow

The analysis is structured into 5 sequential, reproducible Jupyter Notebooks:

```text
├── Yelp_01_Ingestion_Preprocessing.ipynb       # Big data schema filtering & cleaning
├── Yelp_02_EDA_Geospatial_Networks.ipynb       # GIS spatial heatmaps & interaction graphs
├── Yelp_03_NLP_Sentiment_TopicModeling.ipynb   # VADER/Transformers sentiment & LDA topic clustering
├── Yelp_04_Recommender_Systems.ipynb           # Collaborative filtering & Matrix Factorization
└── Yelp_05_Pipeline_Integration_Benchmark.ipynb # End-to-end integration & evaluation metrics
```

---

### 1. Ingestion & Preprocessing (`Yelp_01_Ingestion_Preprocessing.ipynb`)
- Streaming ingestion and schema standardization for high-volume JSON business and review records.
- Handling missing data, type casting, text normalization, and deduplication.

### 2. Exploratory Data Analysis & Geospatial GIS (`Yelp_02_EDA_Geospatial_Networks.ipynb`)
- **Spatial Analytics:** Geospatial density clustering using **Folium** interactive maps and coordinates.
- **Network Graphs:** Analyzing bipartite graphs of customer interactions, check-ins, and business categories with **NetworkX**.

### 3. NLP, Sentiment Analysis & Topic Modeling (`Yelp_03_NLP_Sentiment_TopicModeling.ipynb`)
- Text tokenization, stopword removal, and lemmatization across user reviews.
- Polarity and subjectivity scoring with sentiment analysis.
- Unsupervised **LDA (Latent Dirichlet Allocation)** and **TF-IDF** to uncover latent business themes and dining attributes.

### 4. Hybrid Recommender Systems (`Yelp_04_Recommender_Systems.ipynb`)
- **Collaborative Filtering:** User-Item rating matrix decomposition (SVD / Matrix Factorization).
- **Content-Based Filtering:** Feature similarity using business metadata and extracted review topics.
- Hybrid recommendation scoring predicting personalized ratings with low RMSE.

### 5. Pipeline Integration & Benchmarking (`Yelp_05_Pipeline_Integration_Benchmark.ipynb`)
- Unified evaluation suite comparing baseline recommenders against matrix factorization models.
- Validation metrics: **RMSE**, **MAE**, **Precision@K**, and **Recall@K**.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.10+
- **Data Engineering:** Pandas, NumPy, PyArrow
- **Web Dashboard:** Streamlit
- **Machine Learning & NLP:** Scikit-Learn, NLTK, SciPy
- **Geospatial & Visualization:** Folium, Matplotlib, Seaborn
- **Network Analysis:** NetworkX

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/valeriofiorentini/yelp-bigdata-mining-recommender.git
cd yelp-bigdata-mining-recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
# or with uv / poetry:
uv pip install -r requirements.txt
```

### 3. Data Setup
Download the official Yelp dataset from [Yelp Open Dataset](https://www.yelp.com/dataset) and place the JSON files in a `data/` directory.

### 4. Run Notebooks
Execute notebooks sequentially from `01` to `05` using Jupyter Lab or VS Code.

### 5. Launch the Interactive Dashboard (Streamlit)
A fully interactive web dashboard is included to demonstrate the NLP sentiment analysis, geospatial maps, and recommendation engine in real-time.

```bash
uv run streamlit run app.py
```
*(We highly recommend using `uv` to instantly launch the dashboard in an isolated environment without dependency conflicts).* 

---

## 👤 Author

**Valerio Fiorentini**
- Portfolio: [cv-fiorentini-valerio.vercel.app](https://cv-fiorentini-valerio.vercel.app)
- GitHub: [@valeriofiorentini](https://github.com/valeriofiorentini)
- Email: [valeriofiorentini2002@gmail.com](mailto:valeriofiorentini2002@gmail.com)
