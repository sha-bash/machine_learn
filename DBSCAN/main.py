import ast
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Чтение данных из файла DATA.txt
data = []
with open('DATA.txt', 'r') as file:
    for line in file:
        # Преобразование строки в кортеж координат
        coords = ast.literal_eval(line)
        data.extend(coords)

# Преобразование списка в numpy array
X = np.array(data)

# Реализация DBSCAN
epsilon = 35.0  # размер эпсилон-окрестности
min_samples = 4  # минимальное число объектов для полной эпсилон-окрестности

db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
labels = db.labels_

no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
no_noise = list(labels).count(-1)

# Вывод информации о кластерах
print('Предполагаемое количество кластеров: %d' % no_clusters)

# Генерация scatter plot для обучающих данных с выделением кластеров разными цветами
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Черный используется для шума.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Предполагаемое количество кластеров: %d' % no_clusters)
plt.show()
