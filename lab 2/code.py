import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(52)

LEARNING_RATE = 0.1

# Класс для реализации перцептрона
class SimplePerceptron:
   def __init__(self, input_count: int):
       self.weights = [random.uniform(-1, 1) for _ in range(input_count)]

   def evaluate(self, inputs: list):
       weighted_sum = np.dot(inputs, self.weights)
       return 0 if np.sign(weighted_sum) == -1 else 1

   def train(self, input_set: list, expected_output: list):
       index_list = list(range(len(input_set)))
       correctly_classified = False
       self.weight_history = list()

       while not correctly_classified:
           self.weight_history.append(self.weights.copy())
           correctly_classified = True
           random.shuffle(index_list)

           for idx in index_list:
               x = input_set[idx]
               y = expected_output[idx]
               output = self.evaluate(x)
               if output != y:
                   correctly_classified = False
                   for j in range(0, len(self.weights)):
                       weight_adjustment = x[j] * LEARNING_RATE
                       if y == 0:
                           weight_adjustment *= -1
                       self.weights[j] += weight_adjustment


class XORGate:
   def __init__(self):
       self.perceptrons = [SimplePerceptron(3) for _ in range(3)]
       self.training_history = [list() for _ in range(3)]

   def evaluate(self, inputs: list):
       return self.perceptrons[2].evaluate([1, self.perceptrons[1].evaluate(inputs), self.perceptrons[0].evaluate(inputs)])

   def train(self):
       self.perceptrons[0].train([(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)], [0, 1, 1, 1])  # OR
       self.perceptrons[1].train([(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)], [1, 1, 1, 0])  # NOT AND
       self.perceptrons[2].train([(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)], [0, 0, 0, 1])  # AND
       self.training_history[0] = self.perceptrons[0].weight_history
       self.training_history[1] = self.perceptrons[1].weight_history
       self.training_history[2] = self.perceptrons[2].weight_history

# Создание перцептронов и обучение для различных логических операций
and_perceptron = SimplePerceptron(3)
and_perceptron.train([(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)], [0, 0, 0, 1])

or_perceptron = SimplePerceptron(3)
or_perceptron.train([(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)], [0, 1, 1, 1])

not_perceptron = SimplePerceptron(2)
not_perceptron.train([(1, 0), (1, 1)], [1, 0])

xor_perceptron = XORGate()
xor_perceptron.train()

# Функция для отрисовки графика для логических операций
def plot_logic_function(weights, logic_func=None):
   x_max = 1 + 0.3
   x_min = 0 - 0.3

   y_min = -weights[1] * x_min / weights[2] - weights[0] / weights[2]
   y_max = -weights[1] * x_max / weights[2] - weights[0] / weights[2]

   plt.axis([-0.3, 1.3, -0.3, 1.3])

   if logic_func is not None:
       for y in (0, 1):
           for x in (0, 1):
               if logic_func([1, x, y]) == 1:
                   plt.plot(x, y, "g+")
               else:
                   plt.plot(x, y, "r_")

   plt.plot((x_min, x_max), (y_min, y_max))

# Функция для отрисовки графика для логической операции NOT
def plot_not_logic_function(weights, logic_func=None):
   x_boundary = -weights[0] / weights[1]
   y_min = -0.3
   y_max = 1.3

   plt.axis([-0.3, 1.3, -0.3, 1.3])

   if logic_func is not None:
       for y in (0, 1):
           for x in (0, 1):
               if logic_func([1, x]) == 1:
                   plt.plot(x, y, "g+")
               else:
                   plt.plot(x, y, "r_")
   plt.plot((x_boundary, x_boundary), (y_min, y_max))

# Отрисовка графиков для каждой логической операции

for weights in and_perceptron.weight_history:
   plot_logic_function(weights, and_perceptron.evaluate)
plt.title ("Процесс обучения персептрона")
plt.show()
plot_logic_function(and_perceptron.weights, and_perceptron.evaluate)
plt.title ("Результат обучения")
plt.show()

for weights in or_perceptron.weight_history:
   plot_logic_function(weights, or_perceptron.evaluate)
plt.title ("Процесс обучения персептрона")
plt.show()
plot_logic_function(or_perceptron.weights, or_perceptron.evaluate)
plt.title ("Результат обучения")
plt.show()

for weights in not_perceptron.weight_history:
   plot_not_logic_function(weights, not_perceptron.evaluate)
plt.title ("Процесс обучения персептрона")
plt.show()
plot_not_logic_function(not_perceptron.weights, not_perceptron.evaluate)
plt.title ("Результат обучения")
plt.show()

for perceptron_weights in xor_perceptron.training_history:
   for weights in perceptron_weights:
       plot_logic_function(weights, xor_perceptron.evaluate)
plt.title ("Процесс обучения персептрона")
plt.show()
plot_logic_function(xor_perceptron.perceptrons[0].weights, xor_perceptron.evaluate)
plot_logic_function(xor_perceptron.perceptrons[1].weights, xor_perceptron.evaluate)
plt.title ("Результат обучения")
plt.show()

# Новая сложная логическая функция ¬X1∧X2∧(X4⊕X2∧X3)
def logic_function(x1, x2, x3, x4):
   and_result = and_perceptron.evaluate([1, x2, x3])
   xor_result = xor_perceptron.evaluate([1, x4, and_result])
   and_result2 = and_perceptron.evaluate([1, x1, x2])
   not_result = not_perceptron.evaluate([1, and_result2])
   and_final_result = and_perceptron.evaluate([1, not_result, xor_result])
  
   return and_final_result

# Проверка работы
print("x1 x2 x3 x4  |  y")
for x1 in (0, 1):
   for x2 in (0, 1):
       for x3 in (0, 1):
           for x4 in (0,1):
               print(f"{x1}  {x2}  {x3}  {x4} ", " | ", logic_function(x1, x2, x3, x4))