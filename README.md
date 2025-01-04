# 🚀 CryptoAI - Kryptowalutowy System Predykcyjny

## 📊 O Projekcie
System do analizy i przewidywania cen kryptowalut wykorzystujący uczenie maszynowe, ze szczególnym uwzględnieniem ADA (Cardano).

## 🛠️ Technologie
- Python 3.8+
- TensorFlow 2.x
- Pandas
- Scikit-learn
- MatPlotLib
- CCXT

## 📁 Struktura Projektu
```
crypto-ai/
│
├── data_unprepared/          # Surowe dane
├── data_prepared/            # Przetworzone dane
├── models/                   # Zapisane modele
│
├── data_downloader.py        # Pobieranie danych z Binance
├── crypto_data_merger.py     # Łączenie danych
├── crypto_data_analyzer.py   # Analiza korelacji
├── crypto_model.py           # Model LSTM
└── crypto_tester.py         # Testowanie modelu
```

## ⚙️ Instalacja

```bash
# Klonowanie repozytorium
git clone https://github.com/twoj-username/crypto-ai.git
cd crypto-ai

# Tworzenie wirtualnego środowiska
python -m venv venv
venv\Scripts\activate

# Instalacja zależności
pip install -r requirements.txt
```

## 📝 Plik Requirements.txt

```text
tensorflow>=2.8.0
pandas>=1.4.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
ccxt>=2.0.0
```

## 🚀 Użycie

1. **Pobieranie danych**
```bash
python data_downloader.py
```

2. **Przygotowanie danych**
```bash
python crypto_data_merger.py
```

3. **Analiza**
```bash
python crypto_data_analyzer.py
```

4. **Trenowanie modelu**
```bash
python crypto_model.py
```

5. **Testowanie**
```bash
python crypto_tester.py
```

## 📊 Funkcje

| Moduł | Opis |
|-------|------|
| 

data_downloader.py

 | Pobieranie historycznych danych z Binance |
| 

crypto_data_merger.py

 | Łączenie i czyszczenie danych |
| 

crypto_data_analyzer.py

 | Analiza korelacji między kryptowalutami |
| 

crypto_model.py

 | Model LSTM do przewidywania cen |
| 

crypto_tester.py

 | Walidacja i wizualizacja wyników |

## 🧪 Model AI

- Architektura: Bidirectional LSTM
- Warstwy: 3 warstwy LSTM (400, 200, 100 neuronów)
- Optymalizacja: Adam (lr=0.001)
- Loss: Huber Loss
- Regularyzacja: Dropout (0.3) + BatchNormalization

## 👥 Autorzy
Jakub LiCHUTKO Liszewski  
