import pandas as pd
import os
from glob import glob

def load_and_prepare_data(filepath):
    try:
        df = pd.read_csv(filepath)
        pair_name = os.path.basename(filepath).split('_7years.csv')[0]
        
        df = df.rename(columns={
            'open': f'{pair_name}_open',
            'close': f'{pair_name}_close',
            'volume': f'{pair_name}_volume'
        })
        
        required_cols = ['timestamp'] + [f'{pair_name}_{col}' for col in ['open', 'close', 'volume']]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return df
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

def merge_crypto_data():
    input_folder = 'data_unprepared'
    output_folder = 'data_prepared'
    os.makedirs(output_folder, exist_ok=True)
    
    csv_files = glob(os.path.join(input_folder, '*_7years.csv'))
    
    if not csv_files:
        print("No CSV files found in data_unprepared folder!")
        return
    
    main_df = None
    
    for file in csv_files:
        print(f"Processing {file}...")
        current_df = load_and_prepare_data(file)
        
        if current_df is None:
            continue
            
        if main_df is None:
            main_df = current_df
        else:
            main_df = pd.merge(main_df, current_df, on='timestamp', how='outer')
    
    if main_df is not None:
        # Sort by timestamp
        main_df = main_df.sort_values('timestamp')
        
        # Forward fill missing values (up to 3 steps)
        main_df = main_df.fillna(method='ffill', limit=3)
        
        # Remove rows with any remaining NaN values
        main_df = main_df.dropna()
        
        # Reorder columns to put ADA/USDT first
        cols = main_df.columns.tolist()
        ada_usdt_cols = [col for col in cols if 'ADA_USDT' in col]
        other_cols = [col for col in cols if col not in ada_usdt_cols and col != 'timestamp']
        
        # New column order: timestamp, ADA/USDT columns, other columns
        new_col_order = ['timestamp'] + ada_usdt_cols + other_cols
        main_df = main_df[new_col_order]
        
        # Save merged dataset
        output_file = os.path.join(output_folder, 'merged_crypto_data.csv')
        main_df.to_csv(output_file, index=False)
        print(f"\nData merged successfully. Output saved to: {output_file}")
        print(f"Final dataset shape: {main_df.shape}")
        print("\nColumns in dataset:")
        for col in main_df.columns:
            print(f"- {col}")
    else:
        print("No data to merge!")

if __name__ == "__main__":
    merge_crypto_data()