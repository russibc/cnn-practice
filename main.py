# Feito com o dataset https://www.kaggle.com/iluvchicken/cheetah-jaguar-and-tiger
# Código exemplo: https://keras.io/examples/vision/image_classification_from_scratch/

# importação das bibliotecas a serem utilizadas
# inclusive para realizar as etapas de convolução
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.preprocessing import image

model = Sequential()

model.add(
    Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(
    Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(units=1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

training_generate = ImageDataGenerator(rescale=1/255,
                                       rotation_range=7,
                                       horizontal_flip=True,
                                       shear_range=0.2,
                                       height_shift_range=0.07,
                                       zoom_range=0.2)

gerador_teste = ImageDataGenerator(rescale=1./255)

training_base = training_generate.flow_from_directory('dataset/training_set',
                                                      target_size=(
                                                          64, 64),
                                                      batch_size=32,
                                                      class_mode='binary')

test_base = gerador_teste.flow_from_directory('dataset/test_set',
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='binary')

model.fit_generator(training_base, steps_per_epoch=1800/32,
                    epochs=5, validation_data=test_base,
                    validation_steps=200/32)

model.save('./models')
model.save_weights('./checkpoint.h5')
