# Импорт необходимых библиотек
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

# Настройки
EPOCHS = 128
BATCH_SIZE = 32

# Загрузка данных CIFAR-10
cifar_dataset = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data()

# Стандартизация данных
mean = np.mean(train_images)
stddev = np.std(train_images)
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Функция для построения графиков
def plot_history(history, title):
    plt.figure(figsize=(12,4))
    
    # График ошибки
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title(f'{title} - Loss')
    plt.legend()
    
    # График точности
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.title(f'{title} - Accuracy')
    plt.legend()
    
    plt.show()

# Конфигурация 1
conf1 = Sequential([
    Conv2D(64, (5,5), strides=(2,2), activation='relu', padding='same', input_shape=(32,32,3)),
    Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same'),
    Flatten(),
    Dense(10, activation='softmax')
])
conf1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history1 = conf1.fit(train_images, train_labels, validation_data=(test_images, test_labels),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
plot_history(history1, 'Conf1')

# Конфигурация 2
conf2 = Sequential([
    Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same', input_shape=(32,32,3)),
    Conv2D(16, (2,2), strides=(2,2), activation='relu', padding='same'),
    Flatten(),
    Dense(10, activation='softmax')
])
conf2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history2 = conf2.fit(train_images, train_labels, validation_data=(test_images, test_labels),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
plot_history(history2, 'Conf2')

# Конфигурация 3
conf3 = Sequential([
    Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same', input_shape=(32,32,3)),
    Dropout(0.2),
    Conv2D(16, (2,2), strides=(2,2), activation='relu', padding='same'),
    Dropout(0.2),
    Flatten(),
    Dense(10, activation='softmax')
])
conf3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history3 = conf3.fit(train_images, train_labels, validation_data=(test_images, test_labels),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
plot_history(history3, 'Conf3')

# Конфигурация 4
conf4 = Sequential([
    Conv2D(64, (4,4), strides=(1,1), activation='relu', padding='same', input_shape=(32,32,3)),
    Dropout(0.2),
    Conv2D(64, (2,2), strides=(2,2), activation='relu', padding='same'),
    Dropout(0.2),
    Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same'),
    Dropout(0.2),
    MaxPooling2D((2,2), strides=(2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
conf4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history4 = conf4.fit(train_images, train_labels, validation_data=(test_images, test_labels),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
plot_history(history4, 'Conf4')

# Конфигурация 5
conf5 = Sequential([
    Conv2D(64, (4,4), strides=(1,1), activation='relu', padding='same', input_shape=(32,32,3)),
    Dropout(0.2),
    Conv2D(64, (2,2), strides=(2,2), activation='relu', padding='same'),
    Dropout(0.2),
    Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same'),
    Dropout(0.2),
    Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same'),
    Dropout(0.2),
    MaxPooling2D((2,2), strides=(2,2)),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
conf5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history5 = conf5.fit(train_images, train_labels, validation_data=(test_images, test_labels),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
plot_history(history5, 'Conf5')

# Конфигурация 6
conf6 = Sequential([
    Conv2D(64, (4,4), strides=(1,1), activation='tanh', padding='same', input_shape=(32,32,3)),
    Conv2D(64, (2,2), strides=(2,2), activation='tanh', padding='same'),
    Conv2D(64, (3,3), strides=(1,1), activation='tanh', padding='same'),
    Conv2D(64, (3,3), strides=(1,1), activation='tanh', padding='same'),
    MaxPooling2D((2,2), strides=(2,2)),
    Flatten(),
    Dense(64, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(10, activation='softmax')
])
conf6.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history6 = conf6.fit(train_images, train_labels, validation_data=(test_images, test_labels),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
plot_history(history6, 'Conf6')

# Сбор результатов
results = {
    'Conf1': conf1.evaluate(test_images, test_labels, verbose=0),
    'Conf2': conf2.evaluate(test_images, test_labels, verbose=0),
    'Conf3': conf3.evaluate(test_images, test_labels, verbose=0),
    'Conf4': conf4.evaluate(test_images, test_labels, verbose=0),
    'Conf5': conf5.evaluate(test_images, test_labels, verbose=0),
    'Conf6': conf6.evaluate(test_images, test_labels, verbose=0)
}

# Построение сводных графиков
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.bar(results.keys(), [r[0] for r in results.values()])
plt.title('Test Loss Comparison')
plt.ylabel('Loss')

plt.subplot(1,2,2)
plt.bar(results.keys(), [r[1] for r in results.values()])
plt.title('Test Accuracy Comparison')
plt.ylabel('Accuracy')

plt.show()

# Добавляем предсказания для первых 4 образцов из тестового набора
configs = [conf1, conf2, conf3, conf4, conf5, conf6]
config_names = ['Conf1', 'Conf2', 'Conf3', 'Conf4', 'Conf5', 'Conf6']

# Берем первые 4 образца из тестового набора
sample_images = test_images[:4]
sample_labels = test_labels[:4]

# Преобразуем истинные метки в индексы классов
true_labels = np.argmax(sample_labels, axis=1)

for i, (config, name) in enumerate(zip(configs, config_names)):
    print(f"\n{name}:")
    # Делаем предсказания
    predictions = config.predict(sample_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Вычисляем MSE между предсказанными вероятностями и истинными метками
    mse = np.mean((predictions - sample_labels) ** 2, axis=1) * 1000  # Умножаем на 1000 для масштаба, как в примере
    
    # Выводим результаты для каждого образца
    for j in range(4):
        print(f"Prediction: [{predicted_labels[j]:.6f}] , true value: {true_labels[j]:.1f} , MSE: {mse[j]:.6f}")