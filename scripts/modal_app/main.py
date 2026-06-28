"""
scripts/modal_app/main.py — Main Entrypoint for Modal Serverless Execution

This script defines the Modal App, the cloud container environment (Image),
and the scheduled cron job that runs the daily Agent-B pipeline.
"""

import modal
from datetime import datetime, timezone
import os

# ============================================================
# 1. Define the Cloud Container (Image)
# ============================================================
# We use a Debian slim base image, install system dependencies for compiling
# llama-cpp-python, and install all Python libraries needed.
# Mount the entire scripts/modal_app directory into the remote container 
# so it can find agents.py, inference.py, data_fetcher.py, and db.py
app_dir = os.path.dirname(__file__)

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "cmake", "git", "curl")
    .pip_install(
        "torch==2.1.2",
        "numpy<2",
        "pandas",
        "yfinance",
        "supabase",
        "python-dotenv",
        "huggingface_hub",
        "requests",
        "beautifulsoup4",
        "crewai",
        "litellm",
        "ta",  # Technical Analysis library
    )
    # Install llama-cpp-python with server using pre-compiled CUDA wheels
    .run_commands(
        "pip install 'llama-cpp-python[server]' --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
    )
    .add_local_dir(app_dir, remote_path="/root")
)

# Initialize the Modal App
app = modal.App("agent-b-daily-pipeline", image=image)

# ============================================================
# 2. Daily Pipeline Function
# ============================================================
# This function runs every day at 08:00 UTC. 
# It requires a GPU (T4 is cost-effective) and secrets.
@app.function(
    schedule=modal.Cron("0 8 * * *"),
    gpu="T4",
    timeout=1800,  # 30 mins max execution time
    secrets=[
        modal.Secret.from_name("supabase-credentials"),
        modal.Secret.from_name("huggingface-token"),
    ]
)
def run_daily_pipeline():
    """
    The orchestrator function. Wakes up, coordinates data fetching, 
    sentiment analysis, forecasting, and saves results to DB.
    """
    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting Daily Pipeline...")
    
    # We will import our modules locally inside the function 
    # to avoid errors during Modal app initialization.
    from data_fetcher import fetch_daily_features
    from agents import run_sentiment_analysis
    from inference import predict_horizons
    from db import log_predictions_to_supabase
    
    try:
        # Step 1: Run Sentiment Agents
        print("Running Sentiment Agents...")
        sentiment_score = run_sentiment_analysis()
        print(f"FinGPT Sentiment Score: {sentiment_score:.2f}")
        
        # Step 2: Fetch Live Features
        print("Fetching Live Market Features...")
        features_tensor, raw_prices = fetch_daily_features(sentiment_score)
        
        # Step 3: Run iTransformer Forecasting
        print("Running iTransformer Inference...")
        predictions = predict_horizons(features_tensor)
        
        # Step 3.5: Un-normalize Z-scores back to actual prices
        mu = raw_prices["brent_mean_30d"]
        sigma = raw_prices["brent_std_30d"]
        for key in predictions.keys():
            if key.startswith("pred_") and "volatility" not in key and "crossover" not in key:
                predictions[key] = (predictions[key] * sigma) + mu
        
        # Step 4: Log to Supabase
        print("Logging to Supabase...")
        log_predictions_to_supabase(
            date=datetime.now(timezone.utc),
            actual_close=raw_prices["brent_close"],
            dxy_value=raw_prices["dxy_close"],
            holiday_flag=raw_prices["holiday_flag"],
            sentiment=sentiment_score,
            preds=predictions
        )
        
        print("Daily Pipeline Completed Successfully!")
        
    except Exception as e:
        print(f"PIPELINE FAILED: {str(e)}")
        raise e

# ============================================================
# Local Entrypoint for Testing
# ============================================================
@app.local_entrypoint()
def test_pipeline():
    """Run `modal run scripts/modal_app/main.py` to test locally."""
    print("Triggering test run on Modal cloud...")
    run_daily_pipeline.remote()
