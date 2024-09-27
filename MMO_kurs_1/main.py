import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from datetime import datetime

# Функция потерь
def loss_function(M):
    return np.exp(-M)

# Градиент функции потерь
def gradient(X, y, w):
    M = y * np.dot(X, w)
    return -np.exp(-M) * y * X

# Алгоритм SGD
def sgd(X, y, learning_rate=0.01, max_iter=1000, tol=1e-5):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    loss_history = []

    for i in range(max_iter):
        idx = np.random.randint(n_samples)
        X_i = X[idx]
        y_i = y[idx]
        grad = gradient(X_i, y_i, w)
        w -= learning_rate * grad
        M = y_i * np.dot(X_i, w)
        loss = loss_function(M)
        loss_history.append(loss)
        if i > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            break

    return w, loss_history

# Функция для предсказания меток классов
def predict(X, w):
    return np.sign(np.dot(X, w))

# Функция для вычисления accuracy
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Функция для вычисления среднего эмпирического риска
def empirical_risk(X, y, w):
    M = y * np.dot(X, w)
    return np.mean(loss_function(M))

# Функция для вывода результатов
def print_results(model_name, accuracy, risk, w=None):
    with open("results.txt", "a", encoding='UTF-8') as f:
        f.write(f'Текущее время: {datetime.now()}\n')
        f.write(f"{model_name}:\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n")
        if risk is not None:
            f.write(f"  Средний эмпирический риск: {risk:.4f}\n")
        else:
            f.write("  Средний эмпирический риск: Недоступен\n")
        if w is not None:
            f.write(f"  Веса: {w}\n")
        f.write("\n")

# Загрузка данных
csv_data_filepath = 'datasets/smoke_detection_iot.csv'
data = pd.read_csv(csv_data_filepath)

# Удаление ненужных столбцов
columns_to_drop = ['Unnamed: 0', 'UTC', 'CNT']
data = data.drop(columns_to_drop, axis=1)

# Разделение данных на обучающую и тестовую выборки
target_column = 'Fire Alarm'
X = data.drop(target_column, axis=1).values
y = data[target_column].values

# Балансировка классов
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Нормализация данных
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Обучение модели с использованием SGD
w, loss_history = sgd(x_train, y_train, learning_rate=0.01, max_iter=1000)
y_pred = predict(x_test, w)
acc = accuracy_score(y_test, y_pred)
risk = empirical_risk(x_test, y_test, w)
print_results("SGD", acc, risk, w)

# Функция потерь с L2-регуляризацией
def loss_function_with_l2(M, w, lambda_reg):
    return np.exp(-M) + lambda_reg * np.dot(w, w)

# Градиент функции потерь с L2-регуляризацией
def gradient_with_l2(X, y, w, lambda_reg):
    M = y * np.dot(X, w)
    return -np.exp(-M) * y * X + 2 * lambda_reg * w

# Алгоритм SGD с L2-регуляризацией
def sgd_with_l2(X, y, learning_rate=0.01, max_iter=1000, tol=1e-5, lambda_reg=0.01):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    loss_history = []

    for i in range(max_iter):
        idx = np.random.randint(n_samples)
        X_i = X[idx]
        y_i = y[idx]
        grad = gradient_with_l2(X_i, y_i, w, lambda_reg)
        w -= learning_rate * grad
        M = y_i * np.dot(X_i, w)
        loss = loss_function_with_l2(M, w, lambda_reg)
        loss_history.append(loss)
        if i > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            break

    return w, loss_history

# Обучение модели с использованием SGD с L2-регуляризацией
lambda_reg = 0.01
w_l2, loss_history_l2 = sgd_with_l2(x_train, y_train, learning_rate=0.01, max_iter=1000, lambda_reg=lambda_reg)
y_pred_l2 = predict(x_test, w_l2)
acc_l2 = accuracy_score(y_test, y_pred_l2)
risk_l2 = empirical_risk(x_test, y_test, w_l2)
print_results("SGD с L2-регуляризацией", acc_l2, risk_l2, w_l2)

# Обучение модели SVM с линейным ядром
svm_model = SVC(kernel='linear')
svm_model.fit(x_train, y_train)
y_pred_svm = svm_model.predict(x_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
risk_svm = empirical_risk(x_test, y_test, svm_model.coef_.ravel())
print_results("SVM с линейным ядром", acc_svm, risk_svm, svm_model.coef_.ravel())

# Обучение модели SVM с полиномиальным ядром (степени 2)
svm_poly_model = SVC(kernel='poly', degree=2)
svm_poly_model.fit(x_train, y_train)
y_pred_svm_poly = svm_poly_model.predict(x_test)
acc_svm_poly = accuracy_score(y_test, y_pred_svm_poly)
print_results("SVM с полиномиальным ядром", acc_svm_poly, None)

# Оценка информативности признаков
correlation_matrix = np.corrcoef(x_train, rowvar=False)
eigenvalues = np.linalg.eigvals(correlation_matrix)
print(f"Собственные значения корреляционной матрицы признаков: {eigenvalues}")

# Визуализация результатов
models = ['SGD', 'SGD с L2-регуляризацией', 'SVM с линейным ядром', 'SVM с полиномиальным ядром']
accuracies = [acc, acc_l2, acc_svm, acc_svm_poly]
risks = [risk, risk_l2, risk_svm, None]

# График accuracy
plt.figure(figsize=(10, 5))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.ylabel('Accuracy')
plt.title('Сравнение accuracy для разных моделей')
plt.show()

# График среднего эмпирического риска
plt.figure(figsize=(10, 5))
plt.bar(models[:-1], risks[:-1], color=['blue', 'green', 'red'])
plt.ylabel('Средний эмпирический риск')
plt.title('Сравнение среднего эмпирического риска для разных моделей')
plt.show()

# Визуализация истории потерь
plt.plot(loss_history)
plt.xlabel('Итерация')
plt.ylabel('Функция потерь')
plt.title('История потерь при обучении')
plt.show()

# Визуализация истории потерь с L2-регуляризацией
plt.plot(loss_history_l2)
plt.xlabel('Итерация')
plt.ylabel('Функция потерь с L2-регуляризацией')
plt.title('История потерь при обучении с L2-регуляризацией')
plt.show()

# ROC-кривые для SVM моделей
def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC кривая для {model_name}')
    plt.legend(loc="lower right")
    plt.show()

# SVM с линейным ядром
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_model.decision_function(x_test))
roc_auc_svm = auc(fpr_svm, tpr_svm)
plot_roc_curve(fpr_svm, tpr_svm, roc_auc_svm, "SVM с линейным ядром")

# SVM с полиномиальным ядром
fpr_svm_poly, tpr_svm_poly, _ = roc_curve(y_test, svm_poly_model.decision_function(x_test))
roc_auc_svm_poly = auc(fpr_svm_poly, tpr_svm_poly)
plot_roc_curve(fpr_svm_poly, tpr_svm_poly, roc_auc_svm_poly, "SVM с полиномиальным ядром")