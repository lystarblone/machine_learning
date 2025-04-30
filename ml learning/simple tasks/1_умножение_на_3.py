import numpy as np
import tensorflow as tf
from numpy.f2py.crackfortran import verbose
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Установка сидов для воспроизводимости
tf.random.set_seed(7)

# Обучающие данные
x_train = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=np.float32)
y_train = np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30], dtype=np.float32)

# Создание модели
model = Sequential([
    Dense(1, input_shape=(1,), activation='linear')
])

# Компиляция модели
model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

# Обучение
model.fit(x_train, y_train, epochs=500, verbose=0)

# Тестовые данные
test_input = np.array([[13], [15], [-3], [100]], dtype=np.float32)

# Предсказания
predictions = model.predict(test_input)

# Вывод результатов
print("\nПроверка модели на тестовых данных:")
for i in range(len(test_input)):
    expected = test_input[i][0] * 3
    predicted = predictions[i][0]
    print(f"Число: {test_input[i][0]}, Ожидаемое: {expected}, Предсказано: {predicted:.2f}, Ошибка: {abs(expected - predicted):.2f}")
