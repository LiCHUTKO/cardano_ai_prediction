import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        y = data.iloc[i + seq_length]['ADA_close']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def sprawdz_model(filepath, model_path, rok):
    # Załaduj model
    model = load_model(model_path)
    
    # Załaduj i przetwórz dane
    data = pd.read_csv(filepath)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    
    # Wybierz kolumny cech
    features = data.columns
    
    # Normalizuj dane
    scaler = RobustScaler()
    data[features] = scaler.fit_transform(data[features])
    
    # Ustaw długość sekwencji na 60 dni
    seq_length = 60
    X, y = create_sequences(data[features], seq_length)
    
    # Dokonaj predykcji
    y_pred = model.predict(X)
    
    # Wybierz dane z wybranego roku
    start_date = f'{rok}-01-01'
    end_date = f'{rok}-12-31'
    mask = (data.index >= start_date) & (data.index <= end_date)
    data_year = data.loc[mask]
    
    # Przygotuj dane do wykresu
    y_test_actual = y[mask][seq_length:]
    y_pred_year = y_pred[mask][seq_length:]
    
    # Dopasuj długości danych
    min_length = min(len(y_test_actual), len(y_pred_year))
    y_test_actual = y_test_actual[:min_length]
    y_pred_year = y_pred_year[:min_length]
    data_year_index = data_year.index[seq_length:seq_length + min_length]
    
    # Narysuj wykres
    plt.figure(figsize=(12, 6))
    plt.plot(data_year_index, y_pred_year, label='Przewidywane ceny', alpha=0.5)
    plt.plot(data_year_index, y_test_actual, label='Rzeczywiste ceny')
    plt.title(f'Rzeczywiste vs Przewidywane ceny (Rok {rok})')
    plt.xlabel('Data')
    plt.ylabel('Cena ADA')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# Przykład użycia funkcji
sprawdz_model('data_prepared/merged_crypto_data.csv', 'trained_model.h5', 2022)