# Классификатор дерева решений с набором данных 

В этом репозитории мы реализуем классификатор дерева решений с использованием набора данных Iris из библиотеки scikit-learn.

## Использование
1. Загрузите набор данных Iris с помощью datasets.load_iris().
2. Подготовьте тренировочные данные, выбрав определенные признаки из набора данных.
3. Создайте экземпляр DecisionTreeClassifier с необходимыми параметрами.
4. Обучите классификатор на тренировочных данных с использованием метода fit().
5. Сгенерируйте сетку для визуализации, вызвав функцию get_grid(data).
6. Предскажите метки классов для точек сетки.
7. Постройте границы решений и данные обучения с помощью методов pcolormesh() и scatter().
8. Отобразите график с помощью метода show().