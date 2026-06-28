"""
scripts/modal_app/db.py — Supabase Logging for Agent-B

Connects to Supabase and securely logs the raw market data, FinGPT sentiment,
and the iTransformer forecasting predictions for all horizons.
"""

import os
from supabase import create_client, Client
from datetime import datetime

def get_supabase_client() -> Client:
    """Initializes and returns a Supabase client using Modal Secrets."""
    # These environment variables will be injected by Modal Secrets
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_SECRET_KEY") # Use service role/secret key for backend insertion
    
    if not url or not key:
        raise ValueError("Missing Supabase credentials in environment variables.")
        
    return create_client(url, key)

def log_predictions_to_supabase(
    date: datetime, 
    actual_close: float, 
    dxy_value: float, 
    holiday_flag: int, 
    sentiment: float, 
    preds: dict
):
    """
    Constructs the exact payload matching the agent_b_predictions schema 
    and inserts the row into Supabase.
    """
    supabase = get_supabase_client()
    
    # Format the payload to perfectly match the schema from the screenshot
    payload = {
        "date": date.isoformat(),
        "actual_close": float(actual_close),
        "dxy_value": float(dxy_value),
        "holiday_flag": int(holiday_flag),
        "fingpt_sentiment": float(sentiment),
        
        # 1-Day Horizon
        "pred_1d_10th": preds["pred_1d_10th"],
        "pred_1d_50th": preds["pred_1d_50th"],
        "pred_1d_90th": preds["pred_1d_90th"],
        
        # 1-Month Horizon
        "pred_1m_10th": preds["pred_1m_10th"],
        "pred_1m_50th": preds["pred_1m_50th"],
        "pred_1m_90th": preds["pred_1m_90th"],
        
        # 3-Month Horizon
        "pred_3m_10th": preds["pred_3m_10th"],
        "pred_3m_50th": preds["pred_3m_50th"],
        "pred_3m_90th": preds["pred_3m_90th"],
        
        # Auxiliary Outputs
        "pred_volatility": preds["pred_volatility"],
        "pred_ma_crossover": preds["pred_ma_crossover"]
    }
    
    print(f"Inserting payload into agent_b_predictions...")
    
    response = supabase.table("agent_b_predictions").insert(payload).execute()
    
    if len(response.data) > 0:
        print(f"Success! Prediction logged with ID: {response.data[0].get('id')}")
    else:
        print("Warning: Supabase returned an empty response. Insertion may have failed.")
        
if __name__ == "__main__":
    # Local test payload (will fail if env vars not set locally)
    print("Testing DB connection...")
    try:
        get_supabase_client()
        print("Connected to Supabase.")
    except Exception as e:
        print(f"Failed to connect: {e}")
