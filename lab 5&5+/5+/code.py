import sys  # Импортируем модуль sys для работы с параметрами системы
assert sys.version_info >= (3, 5)  # Проверяем, что версия Python не ниже 3.5

IS_COLAB = "google.colab" in sys.modules  # Проверяем, запущен ли код в Google Colab
IS_KAGGLE = "kaggle_secrets" in sys.modules  # Проверяем, запущен ли код на платформе Kaggle

import sklearn  # Импортируем библиотеку Scikit-learn
assert sklearn.__version__ >= "0.20"  # Проверяем, что версия scikit-learn не ниже 0.20

import numpy as np  # Импортируем библиотеку NumPy для работы с массивами
import os  # Импортируем библиотеку os для работы с операционной системой

np.random.seed(42)  # Устанавливаем фиксированное начальное значение для генератора случайных чисел (для воспроизводимости)

import matplotlib as mpl  # Импортируем библиотеку Matplotlib для визуализации данных
import matplotlib.pyplot as plt  # Импортируем модуль для рисования графиков

mpl.rc('axes', labelsize=14)  # Настройка размеров шрифтов меток на осях графиков
mpl.rc('xtick', labelsize=12)  # Настройка размера шрифта для меток на оси X
mpl.rc('ytick', labelsize=12)  # Настройка размера шрифта для меток на оси Y

PROJECT_ROOT_DIR = "."  # Устанавливаем текущую директорию как корень проекта
CHAPTER_ID = "classification"  # Идентификатор текущей главы проекта
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)  # Путь для сохранения изображений
os.makedirs(IMAGES_PATH, exist_ok=True)  # Создаем директорию для изображений, если она не существует

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):  # Функция для сохранения изображения
  path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)  # Формируем полный путь к файлу изображения
  print("Saving figure", fig_id)  # Выводим сообщение о сохранении изображения
  if tight_layout:  # Если tight_layout=True, применяем его к графику
      plt.tight_layout()
  plt.savefig(path, format=fig_extension, dpi=resolution)  # Сохраняем график в файл

from sklearn.datasets import fetch_openml  # Импортируем функцию для загрузки данных с OpenML
mnist = fetch_openml('mnist_784', version=1, as_frame=False)  # Загружаем датасет MNIST
mnist.keys()  # Выводим ключи, доступные в загруженном датасете

X, y = mnist["data"], mnist["target"]  # Разделяем данные на признаки (X) и целевую переменную (y)
X.shape  # Получаем размерность матрицы данных

y.shape  # Получаем размерность массива меток

28 * 28  # Размер изображения (28x28 пикселей)

import matplotlib as mpl  # Импортируем Matplotlib для рисования
import matplotlib.pyplot as plt  # Импортируем функции для работы с графиками

some_digit = X[17]  # Выбираем 17-й элемент из данных
some_digit_image = some_digit.reshape(28, 28)  # Преобразуем его в изображение размером 28x28
plt.imshow(some_digit_image, cmap=mpl.cm.binary)  # Отображаем изображение
plt.axis("off")  # Отключаем оси
save_fig("some_digit_plot")  # Сохраняем изображение
plt.show()  # Показываем изображение

y[0]  # Получаем метку первого элемента

some_digit = X[31]  # Выбираем 31-й элемент из данных
some_digit_image = some_digit.reshape(28, 28)  # Преобразуем в изображение
plt.imshow(some_digit_image, cmap=mpl.cm.binary)  # Отображаем изображение
plt.axis("off")  # Отключаем оси
plt.show()  # Показываем изображение

y[0]  # Получаем метку первого элемента

some_digit = X[41]  # Выбираем 41-й элемент из данных
some_digit_image = some_digit.reshape(28, 28)  # Преобразуем в изображение
plt.imshow(some_digit_image, cmap=mpl.cm.binary)  # Отображаем изображение
plt.axis("off")  # Отключаем оси
plt.show()  # Показываем изображение

y[0]  # Получаем метку первого элемента

some_digit = X[46]  # Выбираем 46-й элемент из данных
some_digit_image = some_digit.reshape(28, 28)  # Преобразуем в изображение
plt.imshow(some_digit_image, cmap=mpl.cm.binary)  # Отображаем изображение
plt.axis("off")  # Отключаем оси
plt.show()  # Показываем изображение

y[0]  # Получаем метку первого элемента

some_digit = X[55]  # Выбираем 55-й элемент из данных
some_digit_image = some_digit.reshape(28, 28)  # Преобразуем в изображение
plt.imshow(some_digit_image, cmap=mpl.cm.binary)  # Отображаем изображение
plt.axis("off")  # Отключаем оси
plt.show()  # Показываем изображение

y[0]  # Получаем метку первого элемента

y = y.astype(np.uint8)  # Преобразуем метки в тип данных uint8

def plot_digit(data):  # Функция для отображения одной цифры
  image = data.reshape(28, 28)  # Преобразуем данные в изображение 28x28
  plt.imshow(image, cmap=mpl.cm.binary, interpolation="nearest")  # Отображаем изображение
  plt.axis("off")  # Отключаем оси

def plot_digits(instances, images_per_row=10, **options):  # Функция для отображения нескольких цифр
  size = 28  # Размер изображений (28x28)
  images_per_row = min(len(instances), images_per_row)  # Ограничиваем количество изображений в ряду
  n_rows = (len(instances) - 1) // images_per_row + 1  # Вычисляем количество рядов для изображений
  n_empty = n_rows * images_per_row - len(instances)  # Вычисляем, сколько пустых изображений нужно добавить
  padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)  # Добавляем пустые изображения
  image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))  # Формируем сетку изображений
  big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size, images_per_row * size)  # Переставляем оси и получаем большое изображение
  plt.imshow(big_image, cmap=mpl.cm.binary, **options)  # Отображаем сетку изображений
  plt.axis("off")  # Отключаем оси

plt.figure(figsize=(9,9))  # Создаем фигуру для отображения
example_images = X[:100]  # Выбираем первые 100 изображений
plot_digits(example_images, images_per_row=10)  # Отображаем изображения в виде сетки
save_fig("more_digits_plot")  # Сохраняем изображение
plt.show()  # Показываем изображение


import time
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Данные, необходимые для обучения
X, y = mnist["data"], mnist["target"]
x_train,y_train=X,y

# Задаём исходные данные
num = 8
np.random.seed(5)

# Создаём класс
class NeuraNetwork:
   def __init__(self, x, y, t = {0:4, 1:4, 2:4, 3:1}):
       self.y_train = y
       self.x_train = np.insert(x, 0, 1, axis=1)
       self.top = t
       self.learning_rate = 0.001
       self.index_list = list(range(len(self.x_train)))
       self.n_y = [[0]*value for value in self.top.values()]
       self.n_error = [[0]*value for value in self.top.values()]
       self.n_w = []
       self.generate_w()

   def generate_w(self):
       for i in self.top:
           count = len(self.x_train[0]) if i == 0 else self.top[i-1]+ 1
           shape = (self.top[i], count)
           w = np.random.uniform(-1, 1, shape)
           self.n_w.append(w)
          
   def forward_pass(self, x, key = False):
       # Прямой проход
       for i in range(max(self.top)+1):
           if i == 0:
               for j in range(self.top[i]):
                   self.n_y[i][j] = np.tanh(np.dot(self.n_w[i][j], x))
           elif i == (max(self.top)):
               for j in range(self.top[i]):
                   n_inputs = np.array([1.0]+ self.n_y[i-1])
                   z2 = np.dot(self.n_w[i][j], n_inputs)
                   self.n_y[i][j] = 1.0 / (1.0 + np.exp(-z2))
           else:
               for j in range(self.top[i]):
                   self.n_y[i][j] = np.tanh(np.dot(self.n_w[i][j], np.array([1.0] + self.n_y[i-1])))
           if key == True:
               return self.n_y[-1][0]
          
   def backward_pass(self, y):
       # Обратный проход
       if y == str(num):
           error_prime = -(1.0 - self.n_y[-1][0])
       else:
           error_prime = -(0.0 - self.n_y[-1][0])

       derivative = self.derivative_log()
       self.n_error[-1][0] = error_prime * derivative

       for i in reversed(range(max(self.top))):
           for j in range(self.top[i]):
               sum = 0
               for k in range(self.top[i+1]): sum+=self.n_error[i+1][k] * self.n_w[i+1][k][j+1]
               self.n_error[i][j] = sum * self.derivative_tanh(i, j)

   def adjust_weights(self, x):
       # Корректируем веса
       for i in range(max(self.top)+1):
           if i == 0: inputs = x
           else: inputs = np.array([1.0] + self.n_y[i-1])
           for j in range(self.top[i]): self.n_w[i][j] -= self.learning_rate * self.n_error[i][j] * inputs
   def derivative_log(self, i = -1, j=0): return float(self.n_y[i][j]) * (1.0 - float(self.n_y[i][j]))
   def derivative_tanh(self, i, j): return 1.0 - self.n_y[i][j]**2
n = NeuraNetwork(x_train, y_train, {0:3, 1:2, 2:1})
startTime = time.time()

# Процесс обучения
for i in range(1):
   flag = True
   np.random.shuffle(n.index_list)
   for i in n.index_list:
       n.forward_pass(n.x_train[i])
       n.backward_pass(n.y_train[i])
       n.adjust_weights(n.x_train[i])
   for i in range(len(n.x_train)):
       n.forward_pass(n.x_train[i])
   if(((int(n.y_train[i]) != num) and (float(*n.n_y[-1]) >= 0.5)) or ((int(n.y_train[i]) == num) and (float(*n.n_y[-1]) < 0.5))):
       break
endTime = time.time()
totalTime = endTime - startTime
print("Время обучения ", totalTime)

# Проверка
np.random.shuffle(n.index_list)
lst = []
for i in n.index_list:
   out = n.forward_pass(n.x_train[i], key = True)
   if (n.y_train[i] == str(num) and out >= 0.5) or (n.y_train[i] != str(num) and out < 0.5):
       lst.append(1)
   else:
       lst.append(0)
lst = np.asarray(lst)
print("Эффективность = ", (lst.sum() / lst.size) * 100, "%")

startTime = time.time()
fig, axes = plt.subplots(6, 6, figsize=(8, 8))

# Вывод изображения
for i in range(36):
   if lst[n.index_list[i]] >= 0.5 and n.y_train[n.index_list[i]] == str(num):
       sign, color = "+", "green"
   else:
       sign, color = "-", "red"
   row = i//6
   col = i % 6
   number = X[n.index_list[i]].reshape(28, 28)
   ax = axes[row, col]
   ax.imshow(number, cmap=mpl.cm.binary)
   ax.axis("off")
   ax.set_title(f"{n.y_train[n.index_list[i]]} ({sign})", fontsize=12)
   ax.add_patch(plt.Rectangle((0, 0), 27, 27, fill=False, edgecolor=color, linewidth=2))
   ax.title.set_color(color)
plt.tight_layout()
plt.show()