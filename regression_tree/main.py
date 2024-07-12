from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# Создаем данные для обучения модели
x = np.arange(0, 4, 0.01)  # Отсчеты для исходного сигнала
N = len(x)
y_sin = np.cos(5 * x) * np.exp(x) + 5
y = y_sin + np.random.normal(0, 0.5, N)  # Добавляем шум к исходному сигналу

# Создаем и обучаем модель случайного леса
clf = RandomForestRegressor(max_depth=4, n_estimators=5, random_state=1)
clf.fit(x.reshape(-1, 1), y)  # Обучаем на данных
y_pred = clf.predict(x.reshape(-1, 1))  # Делаем предсказания

# Визуализируем результаты
plt.plot(x, y, label="Original Signal")
plt.plot(x, y_pred, label="RF Regression")
plt.grid()
plt.legend()
plt.title('Random Forest Regression')
plt.show()
