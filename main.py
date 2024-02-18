import pandas as pd
import numpy as np

# Wczytanie danych z logów opartych na unix timestamp
log1_unix_data = pd.read_csv(f'logs/log1_unix.csv')
log2_unix_data = pd.read_csv(f'logs/log2_unix.csv')

# Obliczanie korelacji Pearsona między wartościami z obu logów, bazując na unix timestamp
combined_unix_data = pd.merge(log1_unix_data, log2_unix_data, on='Timestamp', suffixes=('_log1', '_log2'))

# Obliczanie korelacji Pearsona
correlation_unix = combined_unix_data['Value_log1'].corr(combined_unix_data['Value_log2'])


def znajdz_silne_korelacje(data, prog_korelacji=0.8):
    wyniki_korelacji = []  # Lista do przechowywania wyników
    liczba_wierszy = len(data)

    # Iteracja przez każdą parę wartości
    for i in range(liczba_wierszy - 1):  # Pomijamy ostatni wiersz, bo nie ma z czym porównać
        for j in range(i + 1, liczba_wierszy):
            # Obliczanie korelacji Pearsona dla pary wartości
            korelacja = \
            np.corrcoef([data['Value_log1'][i], data['Value_log1'][j]], [data['Value_log2'][i], data['Value_log2'][j]])[
                0, 1]

            # Sprawdzanie, czy korelacja przekracza próg
            if abs(korelacja) >= prog_korelacji:
                wyniki_korelacji.append(((i, j), korelacja))  # Zapisanie indeksów wierszy i wartości korelacji

    return wyniki_korelacji


def analiza_opoznien(data, maks_opoznienie=5):
    najlepsze_opoznienie = 0
    najwyzsza_korelacja = 0

    for opoznienie in range(maks_opoznienie + 1):
        # Przesunięcie danych log2 względem log1
        data_przesunieta = data.copy()
        data_przesunieta['Value_log2'] = data_przesunieta['Value_log2'].shift(opoznienie)

        # Usunięcie wierszy z NaN wynikających z przesunięcia
        data_przesunieta.dropna(inplace=True)

        # Obliczenie korelacji dla przesunięcia
        korelacja = data_przesunieta['Value_log1'].corr(data_przesunieta['Value_log2'])

        # Aktualizacja najlepszego opóźnienia, jeśli znaleziono wyższą korelację
        if abs(korelacja) > najwyzsza_korelacja:
            najwyzsza_korelacja = abs(korelacja)
            najlepsze_opoznienie = opoznienie

    return najlepsze_opoznienie, najwyzsza_korelacja


# Interpretacja siły korelacji
def interpretuj_korelacje(correlation):
    if abs(correlation) > 0.8:
        return "Bardzo silna korelacja"
    elif abs(correlation) > 0.6:
        return "Silna korelacja"
    elif abs(correlation) > 0.4:
        return "Umiarkowana korelacja"
    elif abs(correlation) > 0.2:
        return "Słaba korelacja"
    else:
        return "Bardzo słaba lub brak korelacji"




interpretacja = interpretuj_korelacje(correlation_unix)
print(f"Korelacja między log1 a log2: {correlation_unix:.2f} - {interpretacja}")

# Uruchomienie funkcji znajdz_silne_korelacje na połączonych danych
wyniki_korelacji = znajdz_silne_korelacje(combined_unix_data)

# Wydruk wyników analizy
for para, korelacja in wyniki_korelacji:
    print(f"Pary indeksów: {para}, Wartość korelacji: {korelacja}")