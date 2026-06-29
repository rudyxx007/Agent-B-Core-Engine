<div align="center">
  <h1>🛢️ Agent-B: Core Engine</h1>
  <p><em>Autonomous Multi-Agent Intelligence for Corporate Procurement</em></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
  [![CrewAI](https://img.shields.io/badge/CrewAI-Agentic%20Orchestration-FF6A00.svg?style=for-the-badge)](https://www.crewai.com/)
  [![Modal](https://img.shields.io/badge/Modal-Serverless%20GPU-000000.svg?style=for-the-badge)](https://modal.com/)
  [![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E.svg?style=for-the-badge&logo=supabase)](https://supabase.com/)
</div>

<br/>

> **The Problem:** RIL buys billions of dollars of Brent crude oil every year. A buying decision made one week too early—or too late—can cost millions in lost refining margins. Traditional commodity forecasting relies on static scripts, lagging technical indicators, and human bias.
>
> **The Solution (Agent-B):** A 100% autonomous **Multi-Agent AI System**. Instead of rigid code, Agent-B deploys a collaborative crew of specialized AI agents that wake up every morning, reason about global news, analyze 150 days of macro-economic data, and mathematically predict the exact dollar-value risk of buying oil today versus next month.

*Note: This repository contains the **Core Backend Agentic Engine**. To see the stunning Next.js Dark-Mode UI where the agents' work is visualized, visit the [**Agent-B-UI Dashboard**](https://github.com/rudyxx007/Agent-B-UI).*

---

## ⚡ The "Secret Sauce" (Why This is Elite AI Engineering)

Recruiters and Engineers: This is not a static CSV dataset project. This is a fully containerized, serverless, **Agentic Workflow** built for production. 

### 🤖 The Brains: Multi-Agent Orchestration (`CrewAI`)
This system isn't just a linear Python script; it's a team of autonomous AI agents. Powered by `CrewAI` and a `Qwen3.5-4B` local orchestrator, the system dynamically delegates tasks, handles scraping failures, and controls the execution graph. The agents independently reason about what news sources to scrape and how to process them before passing the data to the deep learning models.

### 👁️ The Eyes: Agentic Web Scraping (`ScrapeGraphAI`)
Instead of blindly parsing HTML, Agent-B reads the news like a human. The CrewAI orchestrator deploys ScrapeGraphAI to hunt down the latest Yahoo Finance articles on Brent crude. Using an internal LLM graph, it reads the full context of the articles, intelligently extracting geopolitical risks and supply chain disruptions without losing crucial context.

### 🧠 The Intuition: Financial Semantic Scoring (`FinGPT`)
We don't rely on generic sentiment analyzers like VADER. The scraped news is fed into a specialized financial LLM (`FinGPT-Llama-3-8B-LoRA`). It reads the context and outputs a strict mathematical scalar between `-1.0` (Bearish Panic) and `+1.0` (Bullish Euphoria), perfectly translating human geopolitics into a structured machine-learning tensor.

### 🔮 The Engine: Quantile Time-Series Forecasting (`iTransformer`)
Agent-B doesn't just guess a single price. Using an `iTransformer` architecture trained with **Pinball Loss (Quantile Regression)**, it predicts the *distribution* of future prices. For every 1-Day, 1-Month, and 3-Month horizon, it calculates:
- 📉 **The 10th Percentile:** The Optimal Buying Zone.
- 🎯 **The 50th Percentile:** The Expected Baseline.
- 📈 **The 90th Percentile:** The Maximum Risk Exposure.

### 🛡️ The Memory: Adaptive Rolling Z-Scores
To prevent neural network "amnesia" over decades of data, raw dollar prices are dynamically converted into 30-day rolling Z-Scores before hitting the PyTorch tensor. The backend mathematically reconstructs them back into actual dollar amounts just milliseconds before logging them to the database.

---

## ⚙️ The Daily Agentic Ballet

Hosted on **Modal**, a scheduled Cron job triggers this exact sequence every single morning:

1. **Wake Up:** Serverless GPU container spins up.
2. **Dispatch the Crew:** CrewAI wakes up the agents to scrape and score global news.
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
│       ├── agents.py          # CrewAI orchestrator & Multi-Agent Logic
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

Want to see the agents run in your own terminal? 

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

---

## 📚 Acknowledgments & References

This project is built on the shoulders of giants. We extend our gratitude to the open-source research community:

* **iTransformer:** Developed by researchers at Tsinghua University (ICLR 2024). [Paper](https://arxiv.org/abs/2310.06625) | [GitHub](https://github.com/thuml/iTransformer)
* **FinGPT:** Maintained by the AI4Finance Foundation. [Paper](https://arxiv.org/abs/2306.06031) | [GitHub](https://github.com/AI4Finance-Foundation/FinGPT)
* **Qwen:** Developed by Alibaba Cloud. [Hugging Face](https://huggingface.co/Qwen)
* **CrewAI:** Multi-agent framework developed by João Moura. [GitHub](https://github.com/joaomdmoura/crewAI)
* **ScrapeGraphAI:** LLM-based scraping pipeline. [GitHub](https://github.com/ScrapeGraphAI/Scrapegraph-ai)

<br/>
<div align="center">
  <i>Architected and developed by <a href="https://github.com/rudyxx007">rudyxx007</a> for Jio Platforms Limited</i>
</div>
