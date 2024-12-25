import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score
from scipy.stats import shapiro

def r2_score_custom(SL, Yout, Text):
    iter = len(Yout)
    numerator = 0
    denominator_1 = 0
    for i in range(iter):
        numerator = numerator + (SL[i] - Yout[i, 0]) ** 2
        denominator_1 = denominator_1 + SL[i]
    denominator_2 = 0
    for i in range(iter):
        denominator_2 = denominator_2 + (SL[i] - (denominator_1 / iter)) ** 2
    R2_score_our = 1 - (numerator / denominator_2)
    print('------------', Text, '-------------')
    print('кількість елементів вбірки=', iter)
    print('Коефіцієнт детермінації (ймовірність апроксимації)=', R2_score_our)

    return R2_score_our


def Stat_characteristics(rates, text):
    num = len(rates)
    ms = np.mean(rates)
    ds = np.var(rates)
    scv = math.sqrt(ds)
    print(f"---------------Статистичні характеристики {text}-----------------------")
    print("Кількість елементів вибірки: ", num)
    print("Математичне очікування: ", ms)
    print("Дисперсія: ", ds)
    print("Середньоквадратичне відхилення: ", scv)
    return


data = np.genfromtxt("usd_to_uah_rates.csv", delimiter=",", skip_header=1)
dates = np.arange(len(data))
rates = data[:, 1]  # Курс долара


Stat_characteristics(rates, "результатів парсингу")



best_r2_score = -np.inf
best_degree = 0
best_model = None

# Оптимізація моделі
for degree in range(2, 7):
    coeffs = np.polyfit(dates, rates, degree)
    poly_model = np.poly1d(coeffs)

    # Передбачення
    predicted_rates = poly_model(dates)

    # Обчислення R2
    score = r2_score(rates, predicted_rates)
    print(f"Коефіцієнт детермінації для полінома ступеня {degree}: {score}")

    if score > best_r2_score:
        best_r2_score = score
        best_degree = degree
        best_model = poly_model

print(f"Найкраща модель: поліном ступеня {best_degree} з R2 = {best_r2_score}")
#c = np.polyfit(dates, rates, 5)
#p = np.poly1d(c)
#best_model = p
# Побудова графіку для найкращої моделі
predicted_rates_best = best_model(dates)
plt.figure(figsize=(10, 6))
plt.plot(dates, rates, label="Вихідні дані", color="blue")
plt.plot(dates, predicted_rates_best, label=f"Поліноміальна модель (ступінь={best_degree})", color="red")
plt.legend()
plt.xlabel("День")
plt.ylabel("Курс USD/UAH")
plt.show()

def Plot_extrapol(dates, rates):
    d = np.arange(round(len(data)*1.5))
    predicted = best_model(d)
    plt.figure(figsize=(10, 6))
    plt.plot(dates, rates, label="Вихідні дані", color="blue")
    plt.plot(d, predicted, label=f"Екстраполяція(p = 6)", color="red")
    plt.legend()
    plt.xlabel("День")
    plt.ylabel("Курс USD/UAH")
    plt.show()
    return
Plot_extrapol(dates,rates)
errors = rates - predicted_rates_best

# Гістограма похибок
plt.hist(errors, bins=20, density=True, alpha=0.7, color='blue')
plt.title("Гістограма похибок")
plt.xlabel("Різниця")
plt.ylabel("Частота")
plt.show()

stat, p = shapiro(errors)
print("Статистика Шапіро-Уілка:", stat)
print("p-значення:", p)

if p > 0.05:
    print("Різниця може походити з нормального розподілу (приймаємо H₀).")
else:
    print("Різниця не походить з нормального розподілу (відхиляємо H₀).")


def detect_anomalies(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev

    # Визначення аномалій
    anomalies = [x for x in data if x < lower_bound or x > upper_bound]
    return anomalies, lower_bound, upper_bound

anom_values = np.random.uniform(70, 90, 5)
rates_with_anomalies = np.concatenate((rates, anom_values))

# Виявлення аномалій в оновленому масиві
print("Аномальні значення в даних:", detect_anomalies(rates_with_anomalies)[0])

