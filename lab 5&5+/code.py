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