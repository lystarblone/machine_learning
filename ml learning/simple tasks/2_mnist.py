import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Установка сида для воспроизводимости
tf.random.set_seed(7)

# Загружаем и подготавливаем данные MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализуем данные: масштабируем пиксели от 0 до 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Преобразуем метки в формат one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Создание модели
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Преобразуем 2D изображение в 1D вектор
    Dense(128, activation='relu'),  # Скрытый слой с 128 нейронами
    Dense(10, activation='softmax')  # Выходной слой для 10 классов (цифры от 0 до 9)
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Сводка модели
model.summary()

# Обучение модели
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Тестовая точность: {test_acc:.4f}")

# Прогноз для примера
import matplotlib.pyplot as plt

seven_indices = np.where(np.argmax(y_test, axis=1) == 7)[0][:5]

# Визуализация 5 изображений цифры "7"
plt.figure(figsize=(15, 3))
for i, idx in enumerate(seven_indices):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[idx], cmap=plt.cm.binary)
    plt.title(f"Label: {np.argmax(y_test[idx])}")
    plt.axis('off')
plt.show()

# Визуализация общей картины выборки (сетка 8x8)
plt.figure(figsize=(10, 10))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.title(f"Label: {np.argmax(y_test[i])}", fontsize=8)
    plt.axis('off')
plt.show()

# Прогнозируем класс изображения
predicted = model.predict(np.expand_dims(x_test[0], axis=0))
predicted_class = np.argmax(predicted)
print(f"Предсказанный класс: {predicted_class}")