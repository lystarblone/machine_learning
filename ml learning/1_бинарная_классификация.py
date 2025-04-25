import numpy as np
import matplotlib.pyplot as plt

# Входные данные для обучения
x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
x_train = [x + [1] for x in x_train]  # Добавляем 1 для учета смещения
x_train = np.array(x_train)
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

# Считаем сумму произведений каждой точки на её метку
pt = np.sum([x * y for x, y in zip(x_train, y_train)], axis=0)
# Строим квадратную матрицу
xxt = np.sum([np.outer(x, x) for x in x_train], axis=0)
# Находим веса модели
w = np.dot(pt, np.linalg.inv(xxt))
print(w)

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