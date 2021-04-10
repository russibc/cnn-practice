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

# definindo a RNA
classificador = Sequential()

# primeiro passo da convolução, criar a camada de convolução
# matriz 3x3 para fazer a multiplicação da matriz (imagem original x kernel)
# 32 é a quantidade de filtros, quando faszer a simulação é bom aumentar para 64
# dimensões das imagens 64 x 64 para padronizar (vai converte todas as imagens
# para 64 pixels)
# 3 canais para informar que é rgb
# relu retira parte mais escuras (valores negativos)
classificador.add(
    Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# pega o mapa de característica e normaliza os valores entre 0 e 1
classificador.add(BatchNormalization())

# matriz do max pooling com as principais características matrix 2x2
classificador.add(MaxPooling2D(pool_size=(2, 2)))

# adicionando mais um camada de convolução
classificador.add(
    Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))

# ultimo etapa da convolução
# alterando a estruta de dados de matriz para vetor atráves do flattering
classificador.add(Flatten())

# configuração da rede neural com 2 camadas ocultas
# adicionando a primeira camada oculta com 128 neuroniso 128 características
# cada imagem tem 64x64 pixels de dimensao cada pixel uma característica
classificador.add(Dense(units=128, activation='relu'))

# o dropout para zerar algumas entradas da camada oculta, neste caso 20%
classificador.add(Dropout(0.2))

# criando mais uma camada oculta
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))

# criando a camada de saída da RNA
classificador.add(Dense(units=1, activation='sigmoid'))

# configurando atributos para o treinamento
classificador.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

# carregar os dados de treinamento
# criar a variável para gerar os dados de treinamentos
# utliza o método ImageGenerator para padronizar a estutura de dados da imagens
# a ser processada no treinamento e defini parâtros de normalização, por exemplo
# o atributo de rescale para normalizar as imagens com valores de pixels 1 à 255
gerador_treinamento = ImageDataGenerator(rescale=1/255,
                                         rotation_range=7,
                                         horizontal_flip=True,
                                         shear_range=0.2,
                                         height_shift_range=0.07,
                                         zoom_range=0.2)
gerador_teste = ImageDataGenerator(rescale=1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size=(
                                                               64, 64),
                                                           batch_size=32,
                                                           class_mode='binary')

# carregar os dados para teste
base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size=(64, 64),
                                               batch_size=32,
                                               class_mode='binary')

# fazendo o treinamento
# os parâmetros steps_per_epoch=4000/32,   validation_steps = 1000/32
# e epochs=5 vai fazer o treinamento ser mais rápido e interfere na
# qualidade do treinamento
classificador.fit_generator(base_treinamento, steps_per_epoch=1800/32,
                            epochs=5, validation_data=base_teste,
                            validation_steps=200/32)

# fazendo o teste com uma imagem de hyena
# carregando a image para realizar o teste na variável imagem_teste
imagem_teste = image.load_img('dataset/test_set/hyena/hyena_006_val_resized.jpg',
                              target_size=(64, 64))

# alterar o formato da imagem de teste
imagem_teste = image.img_to_array(imagem_teste)

# ver os valores de cada pixel de image_teste
# normalizando esses valores na escala de 0 - 1
imagem_teste /= 255

# alterando o formato para o tensor flow adicionando mais uma coluna
imagem_teste = np.expand_dims(imagem_teste, axis=0)

# realizado essas configurações já podemos realizar a previsão
previsao = classificador.predict(imagem_teste)

# retornando false para cheetah e verdadeiro para hyena
previsao = (previsao > 0.5)

classificador.save('./models')
classificador.save_weights('./checkpoint.h5')
