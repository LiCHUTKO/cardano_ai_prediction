import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

# 1. Wczytaj dane
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    return data

# 2. Wczytaj dane i przygotuj je
data = load_and_preprocess_data('data_prepared/merged_crypto_data.csv')
features = data.columns

# 3. Normalizacja danych
scaler = RobustScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=features, index=data.index)

try:
    # 4. Przygotuj sekwencję ostatnich 60 godzin (taka sama długość jak przy treningu)
    seq_length = 60
    last_sequence = data_scaled.iloc[-seq_length:].values
    X_pred = last_sequence.reshape(1, seq_length, len(features))

    # 5. Wczytaj model i wykonaj predykcję
    model = load_model('trained_model.h5')
    prediction_scaled = model.predict(X_pred, verbose=0)

    # 6. Przekształć predykcję na rzeczywistą cenę
    prediction_reshaped = np.zeros((1, len(features)))
    prediction_reshaped[0, data.columns.get_loc('ADA_close')] = prediction_scaled[0, 0]
    prediction = scaler.inverse_transform(prediction_reshaped)[0, data.columns.get_loc('ADA_close')]

    # 7. Wyświetl wyniki
    ostatnia_cena = data['ADA_close'].iloc[-1]
    print(f'\nOstatnia znana cena ADA: ${ostatnia_cena:.4f}')
    print(f'Przewidywana cena ADA za godzinę: ${prediction:.4f}')
    print(f'Przewidywana zmiana: {((prediction - ostatnia_cena) / ostatnia_cena * 100):.2f}%')

except Exception as e:
    print(f"Wystąpił błąd: {str(e)}")
    print("\nSprawdź czy:\n1. Plik z modelem istnieje\n2. Format danych jest poprawny\n3. Masz wszystkie wymagane kolumny")