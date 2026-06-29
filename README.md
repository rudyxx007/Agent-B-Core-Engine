# 🛢️ Project Agent-B: Core Engine

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Modal](https://img.shields.io/badge/Modal-Serverless%20GPU-000000.svg)](https://modal.com/)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E.svg)](https://supabase.com/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Agentic%20Orchestration-FF6A00.svg)](https://www.crewai.com/)
[![FinGPT](https://img.shields.io/badge/FinGPT-Financial%20NLP-brightgreen.svg)](https://github.com/AI4Finance-Foundation/FinGPT)

> **Agent-B** is an enterprise-grade, fully automated predictive AI pipeline that forecasts primary and secondary risk metrics for Brent Crude Oil continuous futures contracts (Ticker: `BZ=F`). 

This project was engineered from the strategic perspective of a **Corporate Procurement Officer** at a major oil refiner (e.g., Reliance Industries Limited), aiming to minimize raw material acquisition costs using state-of-the-art Deep Learning, Large Language Models (LLMs), and Agentic frameworks—all built on a **100% zero-cost open-source architecture**.

---

## 🎯 Executive Summary (Business Impact)

Commodity procurement teams traditionally rely on lagging technical indicators and delayed analyst reports. **Agent-B** solves this by autonomously executing a daily intelligence pipeline:
1. **Scrapes** real-time macroeconomic news and geopolitical developments.
2. **Quantifies** the fundamental market sentiment using a fine-tuned financial LLM (`FinGPT`).
3. **Fuses** this sentiment with 150 days of structural market data (DXY, VIX, Holiday Flags, EIA Inventories, Crack Spreads).
4. **Predicts** future price distributions (1-Day, 1-Month, 3-Month horizons) using a cutting-edge multivariate time-series model (`iTransformer`).
5. **Recommends** an actionable procurement strategy (Accelerate, Delay, or Stagger) based on quantitative risk thresholds.

*Note: This repository contains the **Core AI/ML Engine**. The frontend presentation layer can be found here:* [**Agent-B-UI Dashboard**](https://github.com/rudyxx007/Agent-B-UI)

---

## 🧠 Core Machine Learning Innovations

### 1. Quantile Regression (Pinball Loss)
Instead of predicting a single, deterministic point estimate (which is effectively useless for institutional risk management), the `iTransformer` engine was trained using **Pinball Loss**. The model outputs three distinct probability percentiles simultaneously for every time horizon:
* **10th Percentile (Optimal Buying Zone):** The statistical floor; the bullish cost scenario.
* **50th Percentile (Expected Baseline):** The median statistical expectation.
* **90th Percentile (Maximum Risk Exposure):** The statistical ceiling; the worst-case cost spike.

### 2. Adaptive Rolling Z-Score Normalization
To prevent neural network "amnesia" and preserve long-term macro trends without losing scale, raw dollar prices are dynamically converted into 30-day rolling Z-Scores before tensor injection. Post-inference, predictions are mathematically un-normalized back into exact dollar amounts for the procurement dashboard.

### 3. LLM-Driven Feature Engineering (CrewAI & FinGPT)
A local `Qwen3.5-0.8B` orchestrator controls a CrewAI framework to deploy ScrapeGraphAI for context-preserving web scraping of global oil news. The extracted text is scored by `FinGPT-Llama3-8B-LoRA` to generate a bounded scalar sentiment vector [-1.0 to 1.0], perfectly translating unstructured human geopolitics into a structured mathematical tensor.

---

## 🏗️ Architecture & Daily Execution Flow

The entire pipeline is containerized and hosted on **Modal**. A scheduled Cron job triggers the serverless execution every morning:

1. **Agentic Orchestration:** CrewAI orchestrates the pipeline, commanding agents to target `BZ=F` on Yahoo Finance to extract current metadata and news URLs.
2. **Context Scraping:** ScrapeGraphAI navigates to URLs, compressing full-page text into structured JSON abstracts while preserving critical context (e.g., "despite", "however").
3. **Semantic Scoring:** FinGPT evaluates the JSON abstract and outputs a scalar sentiment score.
4. **Tensor Assembly:** Live closing prices (DXY, Brent, VIX) and macro indicators are aggregated, normalized, and concatenated with the FinGPT score into a `[1, 90, 12]` PyTorch tensor.
5. **Inference Engine:** The `iTransformer` model (weights hosted on Hugging Face) executes a forward pass on the tensor, outputting the multi-horizon quantiles, predicted volatility, and Moving Average (MA) crossover flags.
6. **Data Warehouse:** Raw dollar predictions are securely logged to a **Supabase (PostgreSQL)** database.
7. **UI Delivery:** The Next.js dashboard securely queries Supabase to dynamically render the probability cones and procurement recommendations.

---

## 🛠️ Technology Stack

| Domain | Technology / Tool |
| :--- | :--- |
| **Time-Series Forecasting** | `PyTorch`, `iTransformer` (Multivariate Attention) |
| **Agentic Orchestration** | `CrewAI`, `llama-cpp-python` |
| **LLMs (Local / GGUF)** | `Qwen3.5-0.8B` (Orchestrator), `FinGPT-Llama-3-8B` (Scorer) |
| **Cloud Compute & Cron** | `Modal` (Serverless GPU) |
| **Model Registry** | `Hugging Face Hub` |
| **Database** | `Supabase` (PostgreSQL) |

---

## 📁 Repository Structure

```text
Agent-B-Core-Engine/
├── scripts/
│   └── modal_app/             # Production Serverless Code
│       ├── main.py            # Modal Cron entrypoint & pipeline definition
│       ├── agents.py          # CrewAI orchestrator & FinGPT tool definition
│       ├── data_fetcher.py    # Yahoo Finance/FRED scraping & Tensor assembly
│       ├── inference.py       # iTransformer architecture & Hugging Face weight loading
│       └── db.py              # Supabase PostgreSQL insertion logic
├── notebooks/                 # R&D, Walk-Forward Validation, & Model Training
│   ├── agent-b-core.ipynb     # Kaggle training notebook (iTransformer)
│   └── fingpt-scoring.ipynb   # Historical FinGPT sentiment generation
├── .env                       # Secrets (Supabase, Hugging Face) - GitIgnored
└── README.md                  # You are here
```

---

## 🚀 How to Run Locally

### 1. Environment Setup
Create a `.env` file in the root directory:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_SECRET_KEY=your_supabase_service_role_key
HF_TOKEN=your_hugging_face_token
```

### 2. Install Dependencies
Ensure you have `modal` installed and authenticated on your local machine:
```bash
pip install modal
modal setup
```

### 3. Execute the Pipeline
You can trigger the entire cloud pipeline from your local terminal. Modal will spin up the remote container, execute the agents, run the PyTorch inference, and log to your database:
```bash
modal run scripts/modal_app/main.py
```

### 4. Deploy the Cron Job
To permanently deploy the pipeline to run automatically every morning on the cloud:
```bash
modal deploy scripts/modal_app/main.py
```

---

## 🔮 Future Roadmap (Phase 5)

* **Market Regime Detection (Hidden Markov Models):** Integrating HMMs to mathematically classify the market into "Regimes" (e.g., High Volatility Panic vs. Low Volatility Bull), allowing the neural network to dynamically adjust its sensitivity to news sentiment.
* **Volume Tranching Engine:** Moving beyond binary "Accelerate/Delay" recommendations to specific volume splits (e.g., "Procure 30% today, float 70% to 1-Month horizon") based on predicted volatility spreads.
* **Continuous Online Learning:** Implementing an automated daily backward-pass training loop on Modal to fine-tune the model weights with the previous day's actuals before predicting the next day.

---
*Architected and developed by [rudyxx007](https://github.com/rudyxx007).*