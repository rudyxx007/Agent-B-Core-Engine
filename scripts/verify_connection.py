"""
scripts/verify_connection.py — One-Shot Database Connectivity Test

PURPOSE:
    Run this script ONCE after setting up your Supabase project and .env file.
    It performs a full round-trip test to prove everything is wired correctly:

    1. Connects to Supabase using the SECRET key (write access)
    2. INSERTs a dummy prediction row
    3. Reads it back using the PUBLISHABLE key (read-only access)
    4. DELETEs the dummy row to clean up
    5. Prints a clear ✅ PASS or ❌ FAIL result

HOW TO RUN:
    From the project root (Agent-B-Core-Engine/), run:
        python scripts/verify_connection.py

    If you see ✅ at the end, your database is fully operational.
    If you see ❌, the error message will tell you exactly what went wrong.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Add the project root to Python's import path so we can import config/
# and database/ from inside the scripts/ folder.
#
# Without this, Python would say: "ModuleNotFoundError: No module named 'config'"
# because it only looks in the current directory by default.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SUPABASE_URL
from database.db_client import get_client, TABLE_NAME


def main():
    print("=" * 60)
    print("  AGENT-B: Database Connectivity Verification")
    print("=" * 60)
    print(f"\n[TARGET] {SUPABASE_URL}")
    print(f"[TABLE]  {TABLE_NAME}\n")

    # ------------------------------------------------------------------
    # Step 1: INSERT a dummy row using the SECRET key
    # ------------------------------------------------------------------
    print("Step 1/4: Inserting dummy prediction row (SECRET key)...")

    # We use a fixed far-future date so it doesn't collide with real data.
    dummy_data = {
        "date": "2099-12-31T23:59:59+00:00",
        "actual_close": 99.99,
        "dxy_value": 100.00,
        "holiday_flag": 1,
        "fingpt_sentiment": 0.42,
        "pred_1d_10th": 95.00,
        "pred_1d_50th": 99.00,
        "pred_1d_90th": 103.00,
        "pred_1m_10th": 90.00,
        "pred_1m_50th": 98.00,
        "pred_1m_90th": 106.00,
        "pred_3m_10th": 85.00,
        "pred_3m_50th": 97.00,
        "pred_3m_90th": 110.00,
        "pred_volatility": 8.00,
        "pred_ma_crossover": 1,
    }

    try:
        secret_client = get_client(use_secret=True)
        insert_response = (
            secret_client.table(TABLE_NAME).insert(dummy_data).execute()
        )
        inserted_row = insert_response.data[0]
        row_id = inserted_row["id"]
        print(f"   [OK] INSERT successful. Row ID: {row_id}")
    except Exception as e:
        print(f"   [FAIL] INSERT FAILED: {e}")
        print("\n   Possible causes:")
        print("   - SQL schema hasn't been run yet (run it in Supabase SQL Editor)")
        print("   - SUPABASE_SECRET_KEY is wrong in .env")
        print("   - Network connectivity issue")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: SELECT the row using the PUBLISHABLE key (read-only)
    # ------------------------------------------------------------------
    print("\nStep 2/4: Reading back the row (PUBLISHABLE key)...")

    try:
        pub_client = get_client(use_secret=False)
        select_response = (
            pub_client.table(TABLE_NAME)
            .select("*")
            .eq("id", row_id)
            .execute()
        )
        if select_response.data and len(select_response.data) == 1:
            read_row = select_response.data[0]
            print(f"   [OK] SELECT successful. actual_close = {read_row['actual_close']}")
        else:
            print("   [FAIL] SELECT returned unexpected data.")
            print(f"   Response: {select_response.data}")
            sys.exit(1)
    except Exception as e:
        print(f"   [FAIL] SELECT FAILED: {e}")
        print("\n   Possible causes:")
        print("   - RLS policy for SELECT is missing")
        print("   - SUPABASE_PUBLISHABLE_KEY is wrong in .env")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Verify data integrity
    # ------------------------------------------------------------------
    print("\nStep 3/4: Verifying data integrity...")

    checks_passed = True
    for key, expected in dummy_data.items():
        actual = read_row.get(key)
        # Timestamps come back in different formats, so compare loosely
        if key == "date":
            continue
        if actual != expected:
            print(f"   [FAIL] Mismatch on '{key}': expected {expected}, got {actual}")
            checks_passed = False

    if checks_passed:
        print("   [OK] All values match perfectly.")
    else:
        print("   [WARN] Some values didn't match (see above).")

    # ------------------------------------------------------------------
    # Step 4: DELETE the dummy row (cleanup)
    # ------------------------------------------------------------------
    print("\nStep 4/4: Cleaning up dummy row...")

    try:
        # Use the secret client for delete (publishable can't delete via RLS)
        delete_response = (
            secret_client.table(TABLE_NAME)
            .delete()
            .eq("id", row_id)
            .execute()
        )
        print(f"   [OK] DELETE successful. Test row removed.")
    except Exception as e:
        print(f"   [WARN] DELETE failed (non-critical): {e}")
        print("   The dummy row (date=2099-12-31) is still in your DB.")
        print("   You can delete it manually from the Supabase Table Editor.")

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if checks_passed:
        print("  [PASS] PHASE 1 VERIFICATION: ALL SYSTEMS OPERATIONAL")
        print("  Your Supabase database is fully connected and working.")
        print("  Both PUBLISHABLE (read) and SECRET (write) keys are valid.")
    else:
        print("  [WARN] PHASE 1 VERIFICATION: PARTIAL SUCCESS")
        print("  Connection works but some data integrity checks failed.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
