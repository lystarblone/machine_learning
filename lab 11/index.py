import os
import numpy as np
import pretty_midi
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Masking, Dropout
from midiutil import MIDIFile
import random
from sklearn.model_selection import train_test_split

# Параметры пути и файлов
DATASET_DIR = "lab 11/dataset/training"
VALIDATION_DIR = "lab 11/dataset/validation"
MUSIC_NAME = 'music'

# Параметры обработки midi
TIME_RESOLUTION = 10  # Сколько шагов в секунду
MIN_SEQ_LENGTH = 100
MAX_SEQ_LENGTH = 640
PIANO_NOTE_RANGE = (21, 108)  # Диапазон MIDI-нот для фортепиано
MAX_NOTES_PER_STEP = 4  # Максимум одновременных нот на одном временном шаге
PAUSE_VALUE = 0.0  # Значение для паузы (отсутствия ноты)

# Параметры обучения модели
BATCH_SIZE = 32
EPOCHS = 200

# Параметры генерации midi
MAX_GENERATION_DURATION = 75  # Максимальная длительность генерации в секундах
DEFAULT_BPM = 80
STEPS_PER_BEAT = 4  # Шагов на одну долю (для размера 4/4)
NOTE_RANGE = (1, 88)  # Диапазон нот (нормализованный)


class DatasetProcessor:
    """Класс для обработки MIDI-файлов и подготовки датасета."""
    def __init__(self, training_dir, validation_dir, time_resolution, min_seq_length, max_seq_length):
        self.training_dir = training_dir
        self.validation_dir = validation_dir
        self.time_resolution = time_resolution
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length

    def midi_to_sequence(self, midi_file_path):
        """Преобразует MIDI-файл в числовую последовательность."""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file_path)
            time_steps = np.arange(0, midi_data.get_end_time(), 1 / self.time_resolution)
            sequence = []
            piano_notes = midi_data.instruments[0].notes

            for current_time in time_steps:
                active_notes = sorted([
                    note.pitch - PIANO_NOTE_RANGE[0]
                    for note in piano_notes
                    if note.start <= current_time < note.end
                ])[:MAX_NOTES_PER_STEP]
                step_data = active_notes + [PAUSE_VALUE] * (MAX_NOTES_PER_STEP - len(active_notes))
                sequence.append(list(map(int, step_data)))

            processed_sequences = [
                sequence[i:i + self.max_seq_length]
                for i in range(0, len(sequence), self.max_seq_length)
                if len(sequence[i:i + self.max_seq_length]) >= self.min_seq_length
            ]
            return processed_sequences
        except Exception as error:
            print(f"Ошибка обработки файла {os.path.basename(midi_file_path)}: {error}")
            return None

    def process_dataset(self):
        """Обрабатывает все MIDI-файлы в директориях training и validation."""
        all_sequences = []

        if os.path.exists(self.training_dir):
            midi_files_train = [
                file for file in os.listdir(self.training_dir)
                if file.lower().endswith(('.mid', '.midi'))
            ]
            for midi_file in midi_files_train:
                file_path = os.path.join(self.training_dir, midi_file)
                sequences = self.midi_to_sequence(file_path)
                if sequences:
                    all_sequences.extend(sequences)
            print(f"Обработано {len(midi_files_train)} тренировочных MIDI-файлов")

        if os.path.exists(self.validation_dir):
            midi_files_val = [
                file for file in os.listdir(self.validation_dir)
                if file.lower().endswith(('.mid', '.midi'))
            ]
            for midi_file in midi_files_val:
                file_path = os.path.join(self.validation_dir, midi_file)
                sequences = self.midi_to_sequence(file_path)
                if sequences:
                    all_sequences.extend(sequences)
            print(f"Обработано {len(midi_files_val)} валидационных MIDI-файлов")

        final_sequences = []
        for seq in all_sequences:
            truncated_seq = seq[:self.max_seq_length]
            if len(truncated_seq) >= self.min_seq_length:
                padded_seq = np.pad(
                    truncated_seq,
                    ((0, self.max_seq_length - len(truncated_seq)), (0, 0)),
                    mode='constant',
                    constant_values=PAUSE_VALUE
                )
                final_sequences.append(padded_seq)

        print(f"\nКоличество последовательностей: {len(final_sequences)}")
        print(f"Форма данных: {np.array(final_sequences).shape}")
        return np.array(final_sequences, dtype=np.int32)


class DataLoader:
    """Класс для подготовки данных."""
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length

    def prepare_training_data(self, dataset):
        X, y = [], []
        for sequence in dataset:
            X.append(sequence[:-1] / 88.0)
            y.append(sequence[1:] / 88.0)
        return np.array(X), np.array(y)


class MusicModel:
    """Класс для создания, обучения и использования модели."""
    def __init__(self, input_shape=(None, 4)):
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        model = Sequential([
            Masking(mask_value=PAUSE_VALUE, input_shape=input_shape),
            SimpleRNN(128, activation='relu', return_sequences=True),
            Dropout(0.3),
            SimpleRNN(64, activation='relu', return_sequences=True),
            Dropout(0.3),
            Dense(4, activation='linear')
        ])
        return model

    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'accuracy']
        )

    def train(self, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=None):
        self.compile_model()
        self.model.summary()
        print("Model output shape:", self.model.output_shape)
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=2
        )

    def predict_next_step(self, input_seq):
        next_step = self.model.predict(input_seq, verbose=0)[0]
        next_step = np.clip(np.round(next_step * 88), PAUSE_VALUE, NOTE_RANGE[1]).astype(int)
        return next_step


class MusicGenerator:
    """Класс для генерации музыки."""
    def __init__(self, model, bpm=DEFAULT_BPM, steps_per_beat=STEPS_PER_BEAT, max_duration=MAX_GENERATION_DURATION):
        self.model = model
        self.bpm = bpm
        self.steps_per_beat = steps_per_beat
        self.max_duration = max_duration

    def generate_sequence(self, seed_sequence):
        generated_sequence = seed_sequence.copy()
        beats_per_minute = self.bpm
        total_beats = self.max_duration / 60 * beats_per_minute
        max_steps = int(total_beats * self.steps_per_beat)

        for _ in range(max_steps):
            input_seq = generated_sequence[-639:]
            if len(input_seq) < 1:
                input_seq = generated_sequence[:]
            input_seq = input_seq.reshape(1, -1, 4)
            next_step = self.model.predict_next_step(input_seq)
            generated_sequence = np.vstack([generated_sequence, next_step])

        return generated_sequence[len(seed_sequence):max_steps + len(seed_sequence)]


class MidiConverter:
    """Класс для преобразования последовательности в MIDI-файл."""
    def __init__(self, bpm=DEFAULT_BPM, steps_per_beat=STEPS_PER_BEAT):
        self.bpm = bpm
        self.steps_per_beat = steps_per_beat

    def convert_to_midi(self, sequence, output_file):
        midi = MIDIFile(1)
        midi.addTempo(0, 0, self.bpm)
        time_per_step = 1 / self.steps_per_beat
        active_notes = {}
        added_notes = []

        for step_idx, step in enumerate(sequence):
            current_time = step_idx * time_per_step
            current_pitches = [int(note) + 20 for note in step if note > PAUSE_VALUE]

            for pitch in list(active_notes.keys()):
                if pitch not in current_pitches:
                    for start_time in active_notes[pitch]:
                        added_notes.append({
                            'pitch': pitch,
                            'start_time': start_time,
                            'duration': current_time - start_time
                        })
                    del active_notes[pitch]

            for pitch in current_pitches:
                if pitch not in active_notes:
                    active_notes[pitch] = []
                if not active_notes[pitch]:
                    active_notes[pitch].append(current_time)

        for note in added_notes:
            midi.addNote(
                track=0,
                channel=0,
                pitch=note['pitch'],
                time=note['start_time'],
                duration=note['duration'],
                volume=100
            )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "wb") as f:
            midi.writeFile(f)


# Обработка датасета
dataset_processor = DatasetProcessor(
    training_dir=DATASET_DIR,
    validation_dir=VALIDATION_DIR,
    time_resolution=TIME_RESOLUTION,
    min_seq_length=MIN_SEQ_LENGTH,
    max_seq_length=MAX_SEQ_LENGTH
)
dataset = dataset_processor.process_dataset()  # Получаем датасет напрямую

# Подготовка данных
data_loader = DataLoader(MAX_SEQ_LENGTH)
X, y = data_loader.prepare_training_data(dataset)

# Делим данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Создание и обучение модели
music_model = MusicModel()
history = music_model.train(X_train, y_train, EPOCHS, BATCH_SIZE, (X_test, y_test))

# Оценка модели на тестовых данных
test_loss, test_mae, test_acc = music_model.model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test Accuracy: {test_acc:.4f}")

# Генерация музыки
random_index = random.randint(0, len(X_test) - 1)
seed_sequence = X_train[random_index]
generator = MusicGenerator(music_model, DEFAULT_BPM, STEPS_PER_BEAT, MAX_GENERATION_DURATION)
generated_sequence = generator.generate_sequence(seed_sequence)

# Сохранение MIDI-файла
midi_converter = MidiConverter(DEFAULT_BPM, STEPS_PER_BEAT)
midi_converter.convert_to_midi(generated_sequence, f'lab 11/output/{MUSIC_NAME}.mid')
print(f'Музыка успешно сохранена в lab 11/output/{MUSIC_NAME}.mid')