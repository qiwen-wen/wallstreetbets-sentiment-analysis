# 📈 WallStreetBets Stock Heat Predictor

🌐 **[Live Demo](https://qiwen-wen.github.io/wallstreetbets-sentiment-analysis/)**

> Can Reddit predict the stock market? This project mines r/WallStreetBets posts and comments to predict which tickers are most likely to make a large next-day price move — and ranks them with a learned **heat score**.

---

## 🔍 Problem Statement

For each (ticker, day) pair, predict whether a stock will make a **large next-day move (≥5%)** and assign a heat score (probability of a big move), using WSB discussion data combined with historical price signals.

- **Dataset:** 16,645 ticker-days across 78 tickers (June 2023 – March 2025)
- **Base rate of big moves:** 10.9% (class-imbalanced binary classification)

---

## 🧠 Approach

### Data Pipeline
- Scraped and cleaned r/WallStreetBets posts & comments
- Extracted valid stock tickers using regex (`$TSLA`, `GME`) and cross-validated against Yahoo Finance
- Filtered to top 78 most-mentioned tickers, manually removing ambiguous tokens (slang, crypto, pronouns)
- Merged WSB activity with daily OHLCV price data via `yfinance`

### Feature Engineering
- **WSB numeric:** mention count, score sum/mean, unique authors, post fraction
- **Text (TF-IDF):** unigrams + bigrams on aggregated daily post/comment text per ticker
- **Sentiment:** VADER compound score per post, aggregated daily per ticker
- **Price/volume:** daily return, log volume, 1/3/5-day lagged returns, 5/10-day rolling volatility, relative volume anomaly

### Modeling
- Chronological train/val/test split (70% / 15% / 15%) — no data leakage
- Baselines: always-0, always-1, price-only logistic, WSB-numeric-only logistic, TF-IDF text-only logistic
- Final model: **Logistic Regression** with TF-IDF text + all numeric features, tuned via `GridSearchCV` with `TimeSeriesSplit` (5-fold, F1-optimized)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-Sentiment-green)
![yfinance](https://img.shields.io/badge/yfinance-Price%20Data-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-lightblue?logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-purple?logo=plotly)

---

## 📊 Results

| Model | AUC | F1 |
|---|---|---|
| **Final model (tuned TF-IDF + all numeric)** | **0.8046** | **0.4481** |
| Logistic (price + WSB numeric) | 0.5981 | 0.2792 |
| Logistic (TF-IDF text-only) | 0.5218 | 0.2051 |
| Baseline: always 0 | 0.5000 | 0.0000 |

The final model achieves an **AUC of 0.80**, substantially outperforming all baselines. AUC and F1 are prioritized given the class-imbalanced nature of the task (only 10.9% big-move days).

---


## 💡 Key Findings

- **WSB mention volume correlates with next-day volatility** — stocks with 21+ daily mentions show higher average absolute returns than low-mention stocks
- **Text features alone barely outperform random**, but combining them with price momentum and sentiment unlocks meaningful signal
- **The full feature set** (TF-IDF + sentiment + lagged price/volume) is what drives the AUC jump from ~0.52 to **0.80**

---

## 📌 Acknowledgements

- WSB data sourced from Reddit via the Pushshift API
- Price data via [yfinance](https://github.com/ranaroussi/yfinance)
