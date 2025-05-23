import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
np.random.seed(7)
LEARNING_RATE = 0.01
EPOCHS = 5
# укажите пути к расположению обучающих и тестовых данных через
# TRAIN_IMAGE_FILENAME, TRAIN_LABEL_FILENAME,
# TEST_IMAGE_FILENAME и TEST_LABEL_FILENAME
TRAIN_IMAGE_FILENAME = 'data/mnist/train-images.idx3-ubyte'
TRAIN_LABEL_FILENAME = 'data/mnist/train-labels.idx1-ubyte'
TEST_IMAGE_FILENAME = 'data/mnist/t10k-images.idx3-ubyte'
TEST_LABEL_FILENAME = 'data/mnist/t10k-labels.idx1-ubyte'

def read_mnist():
   train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
   train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)
   test_images = idx2numpy.convert_from_file(TEST_IMAGE_FILENAME)
   test_labels = idx2numpy.convert_from_file(TEST_LABEL_FILENAME) 

# добавьте в эту функцию чтение файлов с обучающими
# и тестовыми наборами данных, их метками в train_images,
# train_labels, test_images и test_labels
# добавьте сюда же фрагменты кода 1.2 и 1.3, приведённые ниже

   #2
   x_train = train_images.reshape(60000, 784)
   mean = np.mean(x_train)
   stddev = np.std(x_train)
   x_train = (x_train - mean) / stddev
   x_test = test_images.reshape(10000, 784)
   x_test = (x_test - mean) / stddev

   #3
   y_train = np.zeros((60000, 10))
   y_test = np.zeros((10000, 10))
   for i, y in enumerate(train_labels):
       y_train[i][y] = 1
   for i, y in enumerate(test_labels):
       y_test[i][y] = 1
   return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = read_mnist()
index_list = list(range(len(x_train)))

def layer_w(neuron_count, input_count):
   weights = np.zeros((neuron_count, input_count+1))
   for i in range(neuron_count):
       for j in range(1, (input_count+1)):
           weights[i][j] = np.random.uniform(-0.1, 0.1)
   return weights
# Объявляем матрицы и вектора, представляющие нейроны
hidden_layer_w = layer_w(25, 784)
hidden_layer_y = np.zeros(25)
hidden_layer_error = np.zeros(25)
output_layer_w = layer_w(10, 25)
output_layer_y = np.zeros(10)
output_layer_error = np.zeros(10)

chart_x = []
chart_y_train = []
chart_y_test = []
def show_learning(epoch_no, train_acc, test_acc):
   global chart_x
   global chart_y_train
   global chart_y_test
   print('номер эпохи:', epoch_no, ', точность обучения: ', '%6.4f' % train_acc,', точность тестирования: ', '%6.4f' % test_acc)
   chart_x.append(epoch_no + 1)
   chart_y_train.append(1.0 - train_acc)
   chart_y_test.append(1.0 - test_acc)

def plot_learning():
   plt.plot(chart_x, chart_y_train, 'r-', label='ошибка обучения')
   plt.plot(chart_x, chart_y_test, 'b-', label='ошибка тестирования')
   plt.axis([0, len(chart_x), 0.0, 1.0])
   plt.xlabel('эпохи обучения')
   plt.ylabel('ошибка')
   plt.legend()
   plt.show()

def forward_pass(x):
   global hidden_layer_y
   global output_layer_y
   # Функция активации для нейронов скрытого слоя
   for i, w in enumerate(hidden_layer_w):
       z = np.dot(w, x)
       hidden_layer_y[i] = np.tanh(z)
   hidden_output_array = np.concatenate((np.array([1.0]), hidden_layer_y))
   # Функция активации для нейронов выходного слоя
   for i, w in enumerate(output_layer_w):
       z = np.dot(w, hidden_output_array)
       output_layer_y[i] = 1.0 / (1.0 + np.exp(-z))

def backward_pass(y_truth):
   global hidden_layer_error
   global output_layer_error
   # Обратное распространение ошибки для каждого выходного нейрона
   # и создание массива всех ошибок выходного нейрона.
   for i, y in enumerate(output_layer_y):
       error_prime = -(y_truth[i] - y) # Производная потерь
       derivative = y * (1.0 - y) # Производная логистической ф-ии
       output_layer_error[i] = error_prime * derivative
   for i, y in enumerate(hidden_layer_y):
       # Создание массива весов, соединяющих выход скрытого
       # нейрона i с с нейронами в выходном слое
       error_weights = []
       for w in output_layer_w:
           error_weights.append(w[i+1])
       error_weight_array = np.array(error_weights)
       # Обратное распространение для скрытых нейронов
       derivative = 1.0 - y**2 # производная ф-ии tanh
       weighted_error = np.dot(error_weight_array, output_layer_error)
       hidden_layer_error[i] = weighted_error * derivative

def adjust_weights(x):
   global output_layer_w
   global hidden_layer_w
   for i, error in enumerate(hidden_layer_error):
       hidden_layer_w[i] -= (x * LEARNING_RATE * error) # Обновляем все веса

   hidden_output_array = np.concatenate((np.array([1.0]), hidden_layer_y))

   for i, error in enumerate(output_layer_error):
       output_layer_w[i] -= (hidden_output_array * LEARNING_RATE * error) # Обновляем все веса

# Цикл обучения сети
for i in range(EPOCHS): # Отсчёт эпох
   np.random.shuffle(index_list) # Случайный порядок
   correct_training_results = 0
   for j in index_list: # Обучение на всех примерах
       x = np.concatenate((np.array([1.0]), x_train[j]))
       forward_pass(x)
       if output_layer_y.argmax() == y_train[j].argmax():
           correct_training_results += 1
       backward_pass(y_train[j])
       adjust_weights(x)
   correct_test_results = 0
   for j in range(len(x_test)): # Оценка сети
       x = np.concatenate((np.array([1.0]), x_test[j]))
       forward_pass(x)
       if output_layer_y.argmax() == y_test[j].argmax():
           correct_test_results += 1
   # Вывод на экран прогресса обучения
   show_learning(i, correct_training_results/len(x_train), correct_test_results/len(x_test))

plot_learning() # Вывод графика