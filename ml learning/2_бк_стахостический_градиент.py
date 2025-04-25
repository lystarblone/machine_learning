import numpy as np
import matplotlib.pyplot as plt

# Сигмоидная функция потерь для вычисления ошибки
def loss(w, x, y):
    M = np.dot(w, x) * y  # Скалярное произведение весов и точки, умноженное на метку
    return 2 / (1 + np.exp(M))  # Формула сигмоидной функции потерь

# Производная сигмоидной функции потерь для корректировки весов
def df(w, x, y):
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y  # Формула производной

# Входные данные для обучения
x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
x_train = [x + [1] for x in x_train]  # Добавляем 1 в конец каждого списка для учета смещения
x_train = np.array(x_train)
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

n_train = len(x_train)
w = [0.0, 0.0, 0.0]
nt0 = 0.0009  # Начальный шаг обучения для SGD
lm = 0.01  # Скорость обновления показателя качества Q
N = 500  # Число итераций обучения

# Начальный показатель качества (средняя функция потерь)
Q = np.mean([loss(w, x, y) for x, y in zip(x_train, y_train)])
Q_plot = [Q]

# Цикл обучения с использованием SGD
for i in range(N):
    nt = nt0 * (1 - i / N)  # Уменьшаем шаг обучения с каждой итерацией
    k = np.random.randint(0, n_train - 1)
    ek = loss(w, x_train[k], y_train[k])  # Считаем потери для выбранной точки
    w = w - nt * df(w, x_train[k], y_train[k])  # Обновляем веса по формуле SGD
    Q = lm * ek + (1 - lm) * Q  # Пересчитываем показатель качества
    Q_plot.append(Q)

print(w)
print(Q_plot)

# Создаём линию, которая будет разделять точки на два класса
line_x = list(range(max(x_train[:, 0])))
# Формула: y = -(w[0]/w[1]) * x - w[2]/w[1]
line_y = [-x * w[0] / w[1] - w[2] / w[1] for x in line_x]

x_0 = x_train[y_train == 1]
x_1 = x_train[y_train == -1]

# Рисуем график
plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
plt.plot(line_x, line_y, color='green')

# Устанавливаем границы графика
plt.xlim([0, 45])
plt.ylim([0, 75])
plt.ylabel("длина")
plt.xlabel("ширина")
plt.grid(True)
plt.show()

# График изменения показателя качества Q
plt.figure(figsize=(10, 6))
plt.plot(range(len(Q_plot)), Q_plot, label='Q (Средняя функция потерь)', color='blue')
plt.xlabel('Итерация')
plt.ylabel('Q')
plt.title('Изменение средней функции потерь Q в процессе обучения')
plt.grid(True)
plt.legend()
plt.show()