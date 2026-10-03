# Credit Card Fraud Detection System

An interactive Machine Learning web application designed to detect fraudulent credit card transactions in real time.

## Overview
* **Domain:** Financial Technology / Anomaly Detection
* **Model:** Random Forest Classifier with Cost-Sensitive Class Weighting (`balanced`)
* **Framework:** Streamlit, Scikit-Learn, Pandas, NumPy
* **Core Problem:** Mitigating extreme class imbalance typical in transactional data (~0.5% fraud).

## Features
* Interactive parameter sliders for anonymized PCA transaction signals (`V1` to `V4`).
* Real-time risk probability calculation.
* Visual risk classification (Approved vs. High-Risk Alert).

## How to Run Locally
```bash
git clone [https://github.com/](https://github.com/)<your-username>/credit-card-fraud-detector.git
cd credit-card-fraud-detector
pip install -r requirements.txt
streamlit run app.py
