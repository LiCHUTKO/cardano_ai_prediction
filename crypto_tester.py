import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model #type: ignore
from datetime import datetime, timedelta

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        y = data.iloc[i + seq_length]['ADA_USDT_close']
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

def predict_next_price():
    # 1. Wczytaj dane
    df = pd.read_csv('data_prepared/merged_crypto_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # 2. Przygotuj features
    features = df.columns
    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=features, index=df.index)
    
    # 3. Przygotuj ostatnią sekwencję danych (14 dni * 24 godziny)
    sequence_length = 14 * 24
    last_sequence = df_scaled.iloc[-sequence_length:].values
    X_pred = last_sequence.reshape(1, sequence_length, len(features))
    
    try:
        # 4. Wczytaj model i wykonaj predykcję
        model = load_model('trained_model.h5')
        prediction_scaled = model.predict(X_pred, verbose=0)
        
        # 5. Przekształć predykcję na rzeczywistą cenę
        prediction_reshaped = np.zeros((1, len(features)))
        prediction_reshaped[0, df.columns.get_loc('ADA_USDT_close')] = prediction_scaled[0, 0]
        predicted_price = scaler.inverse_transform(prediction_reshaped)[0, df.columns.get_loc('ADA_USDT_close')]
        
        # 6. Pobierz ostatnią znaną cenę
        last_price = df['ADA_USDT_close'].iloc[-1]
        
        # 7. Oblicz zmianę procentową
        change = ((predicted_price - last_price) / last_price) * 100
        
        # 8. Przygotuj czas
        last_time = df.index[-1]
        next_time = last_time + timedelta(hours=1)
        
        # 9. Wyświetl wyniki
        print("\n=== Przewidywanie ceny ADA/USDT ===")
        print(f"Ostatnia aktualizacja: {last_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"Przewidywanie na: {next_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"\nOstatnia cena: ${last_price:.4f}")
        print(f"Przewidywana cena: ${predicted_price:.4f}")
        print(f"Zmiana: {change:.2f}%")
        
        # 10. Dodaj wskaźnik trendu
        if change > 0:
            print("Trend: 🔼 WZROST")
        elif change < 0:
            print("Trend: 🔽 SPADEK")
        else:
            print("Trend: ➡️ BEZ ZMIAN")
            
    except Exception as e:
        print(f"Wystąpił błąd: {str(e)}")

if __name__ == "__main__":
    predict_next_price()
    sprawdz_model('data_prepared/merged_crypto_data.csv', 'trained_model.h5', 2024)