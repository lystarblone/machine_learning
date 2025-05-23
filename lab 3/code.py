# Импорт библиотеки numpy для работы с массивами и математическими операциями
import numpy as np

# Установка случайного начального состояния для генератора случайных чисел.
# Это позволяет получать одни и те же результаты при каждом запуске программы.
np.random.seed(3)

# Установка константы скорости обучения (шаг изменения весов во время обучения)
LEARNING_RATE = 0.1

# Список индексов для упорядочивания или перемешивания обучающих данных
index_list = [0, 1, 2, 3]

# Входные данные для обучения, где каждый массив представляет отдельный пример (включая "смещение" в виде первого элемента = 1.0)
x_train = [np.array([1.0, -1.0, -1.0]),  # Пример 1
          np.array([1.0, -1.0, 1.0]),   # Пример 2
          np.array([1.0, 1.0, -1.0]),   # Пример 3
          np.array([1.0, 1.0, 1.0])]    # Пример 4

# Ожидаемые результаты (выходы) для каждого соответствующего примера из x_train
y_train = [0.0, 1.0, 1.0, 0.0]

# Функция для создания и инициализации весов нейрона
def neuron_w(input_count):
   # Создание массива весов с нулями. (input_count + 1, потому что добавляется вес смещения)
   weights = np.zeros(input_count + 1)
   # Для каждого входа (кроме смещения) присваиваем случайный вес из диапазона [-1.0, 1.0]
   for i in range(1, (input_count + 1)):
       weights[i] = np.random.uniform(-1.0, 1.0)
   # Возврат массива весов нейрона
   return weights

# Инициализация весов для трех нейронов (два входных и один выходной)
n_w = [neuron_w(2), neuron_w(2), neuron_w(2)]

# Список для хранения выходных значений каждого нейрона (обновляется во время forward_pass)
n_y = [0, 0, 0]

# Список для хранения ошибок (градиентов) каждого нейрона, вычисляемых в backward_pass
n_error = [0, 0, 0]

# Функция для отображения текущих значений весов нейронов (используется для отладки)
def show_learning():
   # Вывод информации о весах каждого нейрона
   print('Current weights:')
   for i, w in enumerate(n_w):
       print('neuron ', i, ': w0 =', '%5.2f' % w[0],  # Вывод веса смещения (w0)
             ', w1 =', '%5.2f' % w[1],               # Вывод первого веса
             ', w2 =', '%5.2f' % w[2])               # Вывод второго веса
   print('----------------')

# Функция прямого прохода через сеть (вычисление выходных значений нейронов)
def forward_pass(x):
   global n_y  # Объявляем глобальную переменную, чтобы изменять ее внутри функции
   # Вычисление выходного значения первого нейрона с использованием функции активации tanh
   n_y[0] = np.tanh(np.dot(n_w[0], x)) 
   # Вычисление выходного значения второго нейрона
   n_y[1] = np.tanh(np.dot(n_w[1], x)) 
   # Создание массива входов для третьего нейрона (включая смещение и выходы двух предыдущих нейронов)
   n2_inputs = np.array([1.0, n_y[0], n_y[1]])
   # Вычисление линейной комбинации входов для третьего нейрона
   z2 = np.dot(n_w[2], n2_inputs) 
   # Применение функции активации сигмоиды для получения выходного значения третьего нейрона
   n_y[2] = 1.0 / (1.0 + np.exp(-z2)) 

# Функция обратного прохода (вычисление ошибок для каждого нейрона)
def backward_pass(y_truth):
   global n_error  # Используем глобальную переменную для изменения значений ошибок нейронов
   # Вычисление разницы между истинным значением и выходом третьего нейрона
   error_prime = -(y_truth - n_y[2]) 
   # Вычисление производной сигмоидальной функции для третьего нейрона
   derivative = n_y[2] * (1.0 - n_y[2]) 
   # Вычисление градиента ошибки для третьего нейрона
   n_error[2] = error_prime * derivative 
   # Вычисление производной tanh для первого нейрона
   derivative = 1.0 - n_y[0]**2 
   # Вычисление градиента ошибки для первого нейрона, используя вес из второго слоя
   n_error[0] = n_w[2][1] * n_error[2] * derivative 
   # Вычисление производной tanh для второго нейрона
   derivative = 1.0 - n_y[1]**2 
   # Вычисление градиента ошибки для второго нейрона
   n_error[1] = n_w[2][2] * n_error[2] * derivative 

# Функция для корректировки весов на основе ошибок (обучение сети)
def adjust_weights(x):
   global n_w  # Глобальные веса нейронов, которые нужно обновить
   # Коррекция весов первого нейрона
   n_w[0] -= (x * LEARNING_RATE * n_error[0]) 
   # Коррекция весов второго нейрона
   n_w[1] -= (x * LEARNING_RATE * n_error[1]) 
   # Формирование входов для третьего нейрона (смещение и выходы двух предыдущих нейронов)
   n2_inputs = np.array([1.0, n_y[0], n_y[1]]) 
   # Коррекция весов третьего нейрона
   n_w[2] -= (n2_inputs * LEARNING_RATE * n_error[2]) 

# Основной цикл обучения сети
all_correct = False  # Флаг, показывающий, сошлась ли сеть
while not all_correct:  # Пока сеть не обучена полностью
   all_correct = True  # Предполагаем, что все обучено правильно
   # Перемешиваем порядок обучающих примеров для случайности
   np.random.shuffle(index_list) 
   for i in index_list:  # Проходим по каждому примеру
       forward_pass(x_train[i])  # Выполняем прямой проход
       backward_pass(y_train[i])  # Выполняем обратный проход (вычисление ошибок)
       adjust_weights(x_train[i])  # Корректируем веса
       show_learning()  # Отображаем текущие веса для отладки
   for i in range(len(x_train)):  # Проверяем, обучена ли сеть
       forward_pass(x_train[i])  # Выполняем прямой проход
       # Печатаем входные данные и результат
       print('x1 =', '%4.1f' % x_train[i][1], ', x2 =',
             '%4.1f' % x_train[i][2], ', y =',
             '%.4f' % n_y[2]) 
       # Проверяем, правильно ли предсказан результат
       if(((y_train[i] < 0.5) and (n_y[2] >= 0.5))
               or ((y_train[i] >= 0.5) and (n_y[2] < 0.5))):
           all_correct = False  # Если хотя бы один пример неверный, продолжаем обучение