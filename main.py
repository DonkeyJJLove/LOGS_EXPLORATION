import pandas as pd
import numpy as np

def wczytaj_dane(sciezka):
    """Wczytuje dane z pliku CSV zawierającego logi."""
    return pd.read_csv(sciezka)

def oblicz_korelacje_pearsona(data1, data2, klucz='Timestamp'):
    """Oblicza korelację Pearsona między dwoma zestawami danych."""
    combined_data = pd.merge(data1, data2, on=klucz, suffixes=('_log1', '_log2'))
    return combined_data['Value_log1'].corr(combined_data['Value_log2'])

def znajdz_silne_korelacje(data, prog_korelacji=0.8):
    """Znajduje i zwraca pary indeksów z silną korelacją Pearsona."""
    wyniki_korelacji = []  # Lista do przechowywania wyników
    liczba_wierszy = len(data)

    for i in range(liczba_wierszy - 1):
        for j in range(i + 1, liczba_wierszy):
            korelacja = np.corrcoef([data['Value_log1'][i], data['Value_log1'][j]], [data['Value_log2'][i], data['Value_log2'][j]])[0, 1]
            if abs(korelacja) >= prog_korelacji:
                wyniki_korelacji.append(((i, j), korelacja))

    return wyniki_korelacji

def analiza_opoznien(data, maks_opoznienie=5):
    """Analizuje i zwraca najlepsze opóźnienie między dwoma zestawami danych dla maksymalizacji korelacji."""
    najlepsze_opoznienie = 0
    najwyzsza_korelacja = 0

    for opoznienie in range(maks_opoznienie + 1):
        data_przesunieta = data.copy()
        data_przesunieta['Value_log2'] = data_przesunieta['Value_log2'].shift(opoznienie)
        data_przesunieta.dropna(inplace=True)
        korelacja = data_przesunieta['Value_log1'].corr(data_przesunieta['Value_log2'])

        if abs(korelacja) > najwyzsza_korelacja:
            najwyzsza_korelacja = abs(korelacja)
            najlepsze_opoznienie = opoznienie

    return najlepsze_opoznienie, najwyzsza_korelacja

def interpretuj_korelacje(korelacja):
    """Interpretuje wartość korelacji Pearsona."""
    if abs(korelacja) > 0.8:
        return "Bardzo silna korelacja"
    elif abs(korelacja) > 0.6:
        return "Silna korelacja"
    elif abs(korelacja) > 0.4:
        return "Umiarkowana korelacja"
    elif abs(korelacja) > 0.2:
        return "Słaba korelacja"
    else:
        return "Bardzo słaba lub brak korelacji"

# Główny blok kodu
if __name__ == '__main__':
    log1_unix_data = wczytaj_dane('logs/log1_unix.csv')
    log2_unix_data = wczytaj_dane('logs/log2_unix.csv')

    correlation_unix = oblicz_korelacje_pearsona(log1_unix_data, log2_unix_data)
    interpretacja = interpretuj_korelacje(correlation_unix)
    print(f"Korelacja między log1 a log2: {correlation_unix:.2f} - {interpretacja}")

    wyniki_korelacji = znajdz_silne_korelacje(pd.merge(log1_unix_data, log2_unix_data, on='Timestamp', suffixes=('_log1', '_log2')))
    for para, korelacja in wyniki_korelacji:
        print(f"Pary indeksów: {para}, Wartość korelacji: {korelacja}")
