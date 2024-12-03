import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_binance_data(symbol, timeframe, since):
    try:
        exchange = ccxt.binance()
        all_data = []
        
        while since < exchange.milliseconds():
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if len(ohlcv) == 0:
                break
            filtered_data = [[candle[0], candle[1], candle[4], candle[5]] for candle in ohlcv]
            all_data.extend(filtered_data)
            since = ohlcv[-1][0] + 1
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except ccxt.BadSymbol:
        print(f"Para {symbol} nie jest dostępna na Binance")
        return pd.DataFrame()

def main():
    data_folder = 'data_unprepared'
    os.makedirs(data_folder, exist_ok=True)
    
    pairs = ['ADA/USDT', 
             'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'SOL/USDT', 'DOT/USDT', 'DOGE/USDT', 
             'ADA/BTC', 'ADA/ETH', 'ADA/BNB', 'ADA/XRP', 'ADA/SOL', 'ADA/DOT', 'ADA/DOGE']
    timeframe = '1h'
    
    for pair in pairs:
        filename = f"{pair.replace('/', '_')}_7years.csv"
        filepath = os.path.join(data_folder, filename)
        print(f"Pobieranie danych dla pary: {pair}")
        
        try:
            if os.path.exists(filepath):
                existing_data = pd.read_csv(filepath)
                last_timestamp = existing_data['timestamp'].max()
                since = int(pd.to_datetime(last_timestamp).timestamp() * 1000) + 1
            else:
                since = int((datetime.now() - timedelta(days=7*365)).timestamp() * 1000)
                existing_data = pd.DataFrame()
            
            new_data = fetch_binance_data(pair, timeframe, since)
            
            if not new_data.empty:
                combined_data = pd.concat([existing_data, new_data], ignore_index=True).drop_duplicates(subset='timestamp')
                combined_data.to_csv(filepath, index=False)
                print(f"Dane dla {pair} zaktualizowane i zapisane do pliku: {filepath}")
            else:
                print(f"Brak nowych danych dla {pair}")
        except Exception as e:
            print(f"Błąd podczas pobierania danych dla {pair}: {str(e)}")
            continue

if __name__ == "__main__":
    main()