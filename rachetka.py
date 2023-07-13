import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, t

integral_errors = np.loadtxt('data.txt')

mean_error = np.mean(integral_errors)
var_error = np.var(integral_errors)
mode_error = np.argmax(np.histogram(integral_errors, bins='auto')[0])
median_error = np.median(integral_errors)
skewness_error = np.mean(((integral_errors - mean_error)/np.sqrt(var_error))**3)
kurtosis_error = np.mean(((integral_errors - mean_error)/np.sqrt(var_error))**4) - 3


print(f"Математическое ожидание: {mean_error:.2f}")
print(f"Дисперсия: {var_error:.2f}")
print(f"Мода: {mode_error:.2f}")
print(f"Медиана: {median_error:.2f}")
print(f"Симметрия: {skewness_error:.2f}")
print(f"Экцесс: {kurtosis_error:.2f}")

#plt.hist(integral_errors, bins=8, alpha=0.5, ec="black")

# Задания 9
stat, p = shapiro(integral_errors)
alpha = 0.05
print(p)
if float(p) > alpha:
    print("Нормальное распределение")
else:
    print("Ненормальное распределение")

mu = np.mean(integral_errors)
sigma = np.sqrt(np.var(integral_errors))
print(f'Оценка параметров методом моментов: mu={mu:.2f}, sigma={sigma:.2f}')

# Оценка параметров методом максимального правдоподобия
mu_mle, sigma_mle = norm.fit(integral_errors)
print(f'Оценка параметров методом максимального правдоподобия: mu={mu_mle:.2f}, sigma={sigma_mle:.2f}')

#plt.show()

#mu, sigma = norm.fit(integral_errors) # ???

# Построение гистограммы данных
plt.hist(integral_errors, bins=8, density=True, alpha=0.5, ec="black", color='grey')

#plt.show()

# Задание 4
# Сортируем ошибки по возрастанию
sorted_errors = np.sort(integral_errors)
# Разбиваем ошибки на серии
series = np.where(sorted_errors[:-1] < sorted_errors[1:])[0] + 1
series_lengths = np.append(series[0], np.diff(series))
# Рассчитываем ожидаемое количество серий и стандартное отклонение для нулевой гипотезы о случайности ошибок
expected_series = (2 * len(sorted_errors) - 1) / 3
sd_series = np.sqrt((16 * len(sorted_errors) - 29) / 90)
# Вычисляем значение статистики критерия серий
statistic = (np.sum(series_lengths - expected_series) / sd_series)
# Вычисляем критическую точку для выбранного уровня значимости
alpha = 0.05
critical_value = norm.ppf(1 - alpha / 2)
# Сравниваем значение статистики критерия серий со значением критической точки
if np.abs(statistic) <= critical_value:
    print("Ошибки, вероятно, случайны (не отвергаем H0)")
else:
    print("Ошибки не случайны (отвергаем H0)")

# Задача 8
# Вычисление среднего значения ошибки и стандартной ошибки среднего
x_bar = np.mean(integral_errors)
se = np.std(integral_errors, ddof=1) / np.sqrt(len(integral_errors))
# Вычисление статистики t и p-значения
t_stat = x_bar / se
p_value = t.sf(np.abs(t_stat), len(integral_errors)-1) * 2
# Вывод результата
print('t-statistic:', t_stat)
print('p-value:', p_value)
if p_value < 0.05:
    print('Отвергаем нулевую гипотезу')
else:
    print('Не удается отвергнуть нулевую гипотезу')

#mu, sigma = norm.fit(integral_errors)
# Построение графика плотности распределения
x = np.linspace(-300, 300, 150)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'r', linewidth=2)

#plt.plot(integral_errors, p, 'b', linewidth = 0.5)

# Настройка внешнего вида графика
plt.title('Распределение ошибки вычисления интеграла')
plt.xlabel('Значение ошибки')
plt.ylabel('Частота')

plt.show()