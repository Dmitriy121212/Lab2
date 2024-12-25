import numpy as np
import math as mt
def Sliding_Window_AV_Detect_sliding_wind (S0, n_Wind):
    iter = len(S0)
    j_Wind=mt.ceil(iter-n_Wind)+1
    S0_Wind=np.zeros((n_Wind))
    Midi = np.zeros(( iter))
    for j in range(j_Wind):
        for i in range(n_Wind):
            l=(j+i)
            S0_Wind[i] = S0[l]
        Midi[l] = np.median(S0_Wind)
    S0_Midi = np.zeros((iter))
    for j in range(iter):
        S0_Midi[j] = Midi[j]
    for j in range(n_Wind):
        S0_Midi[j] = S0[j]
    return S0_Midi
def find_optimal_nwind(S0, min_window=5, max_window_ratio=0.2):
    max_window = int(len(S0) * max_window_ratio)
    window_sizes = range(min_window, max_window + 1)

    variation_list = []

    for window_size in window_sizes:
        num_windows = len(S0) // window_size
        window_variances = []


        for i in range(num_windows):
            window = S0[i * window_size: (i + 1) * window_size]
            window_variances.append(np.var(window))


        variation = np.var(window_variances)
        variation_list.append(variation)


    optimal_nwind = window_sizes[np.argmin(variation_list)]

    return optimal_nwind


data = np.genfromtxt("usd_to_uah_rates.csv", delimiter=",", skip_header=1)
dates = np.arange(len(data))
rates = data[:, 1]  # Курс долара
anom_values = np.random.uniform(70, 90, 2)
rates_with_anomalies = np.concatenate((rates, anom_values))
n_wind = find_optimal_nwind(rates_with_anomalies)
print(max(rates_with_anomalies))
detected_anomalies = Sliding_Window_AV_Detect_sliding_wind(rates_with_anomalies, n_wind)
print(max(detected_anomalies))
print(f"Оптимальний розмір вікна: {n_wind}")