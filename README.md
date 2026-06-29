<div align="center">
  <h1>🛢️ Agent-B: Core Engine</h1>
  <p><em>Autonomous Procurement Intelligence for Reliance Industries Limited</em></p>

<br/>

> **The Problem:** RIL buys billions of dollars of Brent crude oil every year. A buying decision made one week too early, or too late, can cost millions in lost refining margins. Traditional commodity forecasting relies on lagging technical indicators and human bias.
>
> **The Solution (Agent-B):** A 100% automated, zero-cost AI pipeline that wakes up every morning, reads the global news, analyzes 150 days of macro-economic data, and mathematically predicts the exact dollar-value risk of buying oil today versus next month.

*Note: This repository contains the **Core Backend AI/ML Engine**. To see the stunning Next.js Dark-Mode UI where the magic is visualized, visit the [**Agent-B-UI Dashboard**](https://github.com/rudyxx007/Agent-B-UI).*

---

## ⚡ The "Secret Sauce" (Why This Isn't Just Another Jupyter Notebook)

Recruiters and Engineers: This is not a static CSV dataset project. This is a fully containerized, serverless production pipeline.

### 👁️ The Eyes: Agentic Web Scraping (`CrewAI` + `ScrapeGraphAI`)

Instead of just looking at numbers, Agent-B reads the news. A `Qwen3.5-0.8B` LLM orchestrates a CrewAI workflow to hunt down the latest Yahoo Finance articles on Brent crude. It deploys ScrapeGraphAI to read the full context of the articles, parsing out geopolitical risks and supply chain disruptions.

### 🧠 The Intuition: Financial Semantic Scoring (`FinGPT`)

We don't rely on generic sentiment analyzers. The scraped news is fed into a specialized financial LLM (`FinGPT-Llama-3-8B-LoRA`). It reads the context and outputs a strict mathematical scalar between `-1.0` (Bearish Panic) and `+1.0` (Bullish Euphoria), perfectly translating human geopolitics into a structured machine-learning tensor.

### 🔮 The Engine: Quantile Time-Series Forecasting (`iTransformer`)

Agent-B doesn't just guess a single price. Using an `iTransformer` architecture trained with **Pinball Loss (Quantile Regression)**, it predicts the *distribution* of future prices. For every 1-Day, 1-Month, and 3-Month horizon, it calculates:

- 📉 **The 10th Percentile:** The Optimal Buying Zone.
- 🎯 **The 50th Percentile:** The Expected Baseline.
- 📈 **The 90th Percentile:** The Maximum Risk Exposure.

### 🛡️ The Memory: Adaptive Rolling Z-Scores

To prevent neural network "amnesia" over decades of data, raw dollar prices are dynamically converted into 30-day rolling Z-Scores before hitting the PyTorch tensor. The backend mathematically reconstructs them back into actual dollar amounts just milliseconds before logging them to the database.

---

## ⚙️ The Daily Serverless Ballet

Hosted on **Modal**, a scheduled Cron job triggers this exact sequence every single morning:

1. **Wake Up:** Serverless GPU container spins up.
2. **Read the Room:** Agents scrape and score global news (FinGPT).
3. **Crunch the Numbers:** Live closing prices (DXY, Brent, VIX) and macro indicators are aggregated and normalized into a `[1, 90, 12]` PyTorch tensor.
4. **See the Future:** The `iTransformer` model (pulled from Hugging Face) runs the forward pass.
5. **Log the Truth:** Predictions are un-normalized and pushed to **Supabase (PostgreSQL)**.
6. **Sleep:** Container gracefully shuts down, costing exactly $0.00 while idle.

---

## 📂 Codebase Geography

```text
Agent-B-Core-Engine/
├── scripts/
│   └── modal_app/             # The Production Cloud Environment
│       ├── main.py            # Cron entrypoint & pipeline definition
│       ├── agents.py          # CrewAI orchestrator & FinGPT logic
│       ├── data_fetcher.py    # Yahoo/FRED scraping & Tensor assembly
│       ├── inference.py       # iTransformer architecture & HF loading
│       └── db.py              # Supabase database insertion
├── notebooks/                 # The R&D Lab
│   ├── agent-b-core.ipynb     # Kaggle dual-T4 training & Walk-Forward Validation
│   └── fingpt-scoring.ipynb   # Historical LLM sentiment backtesting
└── .env                       # Secrets (GitIgnored)
```

---

## 🚀 Spin It Up Locally

Want to see the pipeline run in your own terminal?

**1. Clone & Env Setup**

```bash
git clone https://github.com/rudyxx007/Agent-B-Core-Engine.git
cd Agent-B-Core-Engine
# Create a .env file with SUPABASE_URL, SUPABASE_SECRET_KEY, and HF_TOKEN
```

**2. Modal Auth**

```bash
pip install modal
modal setup
```

**3. Run the Daily Pipeline (Test Mode)**

```bash
modal run scripts/modal_app/main.py
```

**4. Deploy the Cloud Cron Job**

```bash
modal deploy scripts/modal_app/main.py
```

---

## 🔭 What's Next? (Phase 5)

* **Market Regime Detection:** Integrating Hidden Markov Models (HMMs) to classify the market into "Regimes" (e.g., Panic vs. Bull), teaching the neural network to dynamically adjust its sensitivity.
* **Volume Tranching:** Moving beyond binary "Buy/Wait" recommendations to suggest specific volume splits (e.g., "Procure 30% today, float 70% to 1-Month").
* **Online Learning:** Adding an automated daily backward-pass training loop to fine-tune the model weights on the cloud before predicting the next day.

<br/>
<div align="center">
  <i>Architected and developed by <a href="https://github.com/rudyxx007">rudyxx007</a> for Jio Platforms Limited</i>
</div>
