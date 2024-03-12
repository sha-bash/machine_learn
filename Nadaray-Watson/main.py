import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 4, 0.01)           # отсчеты для исходного сигнала
x_est = np.arange(0, 4, 0.01)      # отсчеты, где производится восстановление функции
N = len(x)
y_sin = np.cos(5 * x) * np.exp(x) + 5
y = y_sin + np.random.normal(0, 0.5, N)

# аппроксимация ядерным сглаживанием
h = float(input('Введите ширину окна: '))
if h <= 0.1:
    h = 0.2
    print('Введено недопустимое значение h')
    print('При окне меньше 0.1 для финитных ядер будут ошибки')
    print('Система автоматически увеличило значение до минимально допустимого')
        
change_kernel = int(input('Выберите ядро Вашей функции: 1=Гауссовское, 2=треугольное, 3=прямоугольное: '))
kernel_name = {1: "Гауссовское", 2: "Треугольное", 3: "Прямоугольное"}
 
def select_kernel(r):
    if change_kernel == 1:
        return np.exp(-2 * r**2)  # гауссовское ядро
    elif change_kernel == 2:
        return np.abs(1 - r) * (r <= 1)  # треугольное ядро
    elif change_kernel == 3:
        return (r <= 1)  # прямоугольное ядро
    else:
        print('Не выбрано ни одного ядра')
        return None

ro = lambda xx, xi: np.abs(xx - xi)  # метрика
w = lambda xx, xi: select_kernel(ro(xx, xi) / h) if select_kernel(ro(xx, xi) / h) is not None else 0

plt.figure(figsize=(14, 14))
plot_number = 0

for h in [0.1, 0.3, 1, 10]:
    y_est = []
    for xx in x_est:
        ww = np.array([w(xx, xi) for xi in x])
        yy = np.dot(ww, y) / sum(ww)            # формула Надарая-Ватсона
        y_est.append(yy)

    plot_number += 1
    plt.subplot(2, 2, plot_number)

    plt.scatter(x, y, color='black', s=10)
    plt.plot(x, y_sin, color='blue')
    plt.plot(x_est, y_est, color='red')
    plt.title(f"{kernel_name[change_kernel]} ядро с h = {h}")
    plt.grid()

plt.show()