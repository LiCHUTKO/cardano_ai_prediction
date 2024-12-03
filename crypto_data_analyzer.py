import matplotlib.pyplot as plt
import pandas as pd

def load_and_preprocess_data(filepath):
    """Load and preprocess the cryptocurrency data."""
    data = pd.read_csv(filepath)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    numeric_data = data.select_dtypes(include=[float, int]).fillna(0)
    return numeric_data

def create_correlation_plots(numeric_data):
    """Create correlation scatter plots comparing all columns with ADA_close."""
    target_col = 'ADA_USDT_close'
    other_cols = [col for col in numeric_data.columns if col != target_col]
    n_cols = len(other_cols)
    n_rows = (n_cols + 3) // 4
    
    fig = plt.figure(figsize=(20, 5 * n_rows))
    plot_count = 1
    
    for col in other_cols:
        plt.subplot(n_rows, 4, plot_count)
        plt.scatter(numeric_data[col], numeric_data[target_col], alpha=0.5)
        plt.xlabel(col)
        plt.ylabel('ADA Price')
        plt.title(f'ADA vs {col}')
        plot_count += 1
    
    return fig

def save_and_display_plots(fig, output_path):
    """Save and display the correlation plots."""
    fig.tight_layout()
    plt.savefig(output_path)
    plt.show()

def main():
    """Main function to analyze cryptocurrency data."""
    # Load and preprocess data
    numeric_data = load_and_preprocess_data('data_prepared/merged_crypto_data.csv')
    
    # Create correlation plots comparing each column with ADA
    fig = create_correlation_plots(numeric_data)
    
    # Save and display results
    save_and_display_plots(fig, 'ada_correlations_scatter.png')

if __name__ == "__main__":
    main()