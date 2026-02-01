#!/usr/bin/env python3
"""
Fetch Stock Connect (Southbound) data from HKEX
Southbound = Mainland money flowing INTO Hong Kong stocks
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"


def fetch_stock_connect_daily(date_str: str) -> dict:
    """
    Fetch Stock Connect data for a specific date.
    
    Args:
        date_str: Date in YYYYMMDD format
    
    Returns:
        dict with southbound buy/sell/net in millions HKD
    """
    url = f"https://www.hkex.com.hk/eng/csm/DailyStat/data_tab_daily_{date_str}e.js"
    
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        
        content = resp.text
        content = content.replace('tabData = ', '').rstrip().rstrip(';')
        data = json.loads(content)
        
        result = {
            'date': None,
            'sse_southbound_buy': 0,
            'sse_southbound_sell': 0,
            'szse_southbound_buy': 0,
            'szse_southbound_sell': 0,
        }
        
        for item in data:
            market = item.get('market', '')
            result['date'] = item.get('date', '')
            
            # Only get Southbound (money into HK)
            if 'Southbound' not in market:
                continue
            
            for content_item in item.get('content', []):
                table = content_item.get('table', {})
                schema = table.get('schema', [[]])[0]
                tr = table.get('tr', [])
                
                # Find the summary table with Buy/Sell Turnover
                if 'Buy Turnover' in schema and 'Sell Turnover' in schema:
                    buy_idx = schema.index('Buy Turnover')
                    sell_idx = schema.index('Sell Turnover')
                    
                    if len(tr) > max(buy_idx, sell_idx):
                        buy_val = tr[buy_idx].get('td', [['']])[0][0]
                        sell_val = tr[sell_idx].get('td', [['']])[0][0]
                        
                        # Clean values (remove commas), values are in millions
                        buy_val = float(buy_val.replace(',', '')) if buy_val else 0
                        sell_val = float(sell_val.replace(',', '')) if sell_val else 0
                        
                        if 'SSE' in market:
                            result['sse_southbound_buy'] = buy_val
                            result['sse_southbound_sell'] = sell_val
                        elif 'SZSE' in market:
                            result['szse_southbound_buy'] = buy_val
                            result['szse_southbound_sell'] = sell_val
                    break  # Found the summary table
        
        # Calculate totals
        result['total_southbound_buy'] = result['sse_southbound_buy'] + result['szse_southbound_buy']
        result['total_southbound_sell'] = result['sse_southbound_sell'] + result['szse_southbound_sell']
        result['southbound_net'] = result['total_southbound_buy'] - result['total_southbound_sell']
        
        return result
    except Exception as e:
        print(f"Error fetching {date_str}: {e}")
        return None


def fetch_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical Stock Connect data.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        DataFrame with daily southbound flow data
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    results = []
    current = start
    
    while current <= end:
        # Skip weekends
        if current.weekday() < 5:
            date_str = current.strftime("%Y%m%d")
            print(f"Fetching {date_str}...", end=" ")
            
            data = fetch_stock_connect_daily(date_str)
            if data and data['date']:
                results.append(data)
                print(f"Net: {data['southbound_net']:+,.0f}M HKD")
            else:
                print("No data")
            
            time.sleep(0.3)  # Be nice to HKEX server
        
        current += timedelta(days=1)
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    return df


def main():
    """Fetch and save Stock Connect data."""
    # Fetch last 3 years
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
    
    print(f"Fetching Stock Connect data from {start_date} to {end_date}")
    print("=" * 60)
    
    df = fetch_historical_data(start_date, end_date)
    
    if len(df) > 0:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DATA_DIR / "stock_connect.csv"
        df.to_csv(output_path)
        print(f"\nSaved {len(df)} records to {output_path}")
        
        print(f"\nSummary:")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Avg daily net flow: {df['southbound_net'].mean():,.0f}M HKD")
        print(f"  Max inflow: {df['southbound_net'].max():,.0f}M HKD")
        print(f"  Max outflow: {df['southbound_net'].min():,.0f}M HKD")
    else:
        print("No data fetched")


if __name__ == "__main__":
    main()
