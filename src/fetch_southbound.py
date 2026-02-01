#!/usr/bin/env python3
"""
Fetch Southbound (港股通) historical data from East Money API.
Southbound = Mainland money flowing INTO Hong Kong stocks.

Data types:
- MUTUAL_TYPE=002: 港股通(沪) Shanghai → HK
- MUTUAL_TYPE=004: 港股通(深) Shenzhen → HK
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"


def fetch_southbound_data(pages: int = 250) -> pd.DataFrame:
    """
    Fetch all historical Southbound data from East Money.
    
    Args:
        pages: Number of pages to fetch (20 records per page)
    
    Returns:
        DataFrame with daily southbound flow data
    """
    base_url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://data.eastmoney.com/hsgt/index.html"
    }
    
    all_data = []
    
    # Fetch both Shanghai and Shenzhen southbound
    for mutual_type, name in [("002", "SH_Southbound"), ("004", "SZ_Southbound")]:
        print(f"\nFetching {name} (type={mutual_type})...")
        
        for page in range(1, pages + 1):
            params = {
                "reportName": "RPT_MUTUAL_DEAL_HISTORY",
                "columns": "ALL",
                "filter": f'(MUTUAL_TYPE="{mutual_type}")',
                "pageNumber": page,
                "pageSize": 20,
                "sortColumns": "TRADE_DATE",
                "sortTypes": "-1"
            }
            
            try:
                resp = requests.get(base_url, params=params, headers=headers, timeout=10)
                data = resp.json()
                
                if not data.get("result") or not data["result"].get("data"):
                    print(f"  No more data at page {page}")
                    break
                
                records = data["result"]["data"]
                if not records:
                    break
                
                for rec in records:
                    all_data.append({
                        "date": rec.get("TRADE_DATE", "")[:10],
                        "type": name,
                        "buy_amt": rec.get("BUY_AMT"),
                        "sell_amt": rec.get("SELL_AMT"),
                        "net_amt": rec.get("NET_DEAL_AMT"),
                        "deal_amt": rec.get("DEAL_AMT"),
                    })
                
                if page % 50 == 0:
                    print(f"  Page {page}... {records[0].get('TRADE_DATE', '')[:10]}")
                
                time.sleep(0.1)  # Be nice to the server
                
            except Exception as e:
                print(f"  Error at page {page}: {e}")
                break
    
    if not all_data:
        print("No data fetched!")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["date"])
    
    # Pivot to get SH and SZ in same row
    sh = df[df["type"] == "SH_Southbound"].set_index("date")[["buy_amt", "sell_amt", "net_amt"]]
    sh.columns = ["sh_buy", "sh_sell", "sh_net"]
    
    sz = df[df["type"] == "SZ_Southbound"].set_index("date")[["buy_amt", "sell_amt", "net_amt"]]
    sz.columns = ["sz_buy", "sz_sell", "sz_net"]
    
    merged = sh.join(sz, how="outer")
    
    # Calculate totals
    merged["southbound_buy"] = merged["sh_buy"].fillna(0) + merged["sz_buy"].fillna(0)
    merged["southbound_sell"] = merged["sh_sell"].fillna(0) + merged["sz_sell"].fillna(0)
    merged["southbound_net"] = merged["sh_net"].fillna(0) + merged["sz_net"].fillna(0)
    
    merged = merged.sort_index()
    
    return merged


def main():
    """Fetch and save Southbound data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Fetching Southbound (港股通) data from East Money...")
    print("=" * 60)
    
    df = fetch_southbound_data(pages=250)
    
    if len(df) > 0:
        output_path = DATA_DIR / "southbound.csv"
        df.to_csv(output_path)
        print(f"\n✅ Saved {len(df)} records to {output_path}")
        
        print(f"\n📊 Summary:")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Total records: {len(df)}")
        print(f"  Avg daily net flow: {df['southbound_net'].mean():,.0f}M CNY")
        print(f"  Max inflow:  {df['southbound_net'].max():,.0f}M CNY")
        print(f"  Max outflow: {df['southbound_net'].min():,.0f}M CNY")
        
        # Recent data
        print(f"\n📅 Last 5 days:")
        for date, row in df.tail(5).iterrows():
            print(f"  {date.date()}: Net {row['southbound_net']:+,.0f}M CNY")
    else:
        print("❌ No data fetched")


if __name__ == "__main__":
    main()
