import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QFileInfo
from PyQt5.QtWidgets import QFileDialog

# Feito com o dataset https://www.kaggle.com/iluvchicken/cheetah-jaguar-and-tiger
# Código exemplo: https://keras.io/examples/vision/image_classification_from_scratch/


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(295, 387)
        self.txtPath1 = QtWidgets.QTextEdit(Dialog)
        self.txtPath1.setGeometry(QtCore.QRect(10, 40, 231, 31))
        self.txtPath1.setObjectName("txtPath1")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(10, 90, 131, 19))
        self.label_2.setObjectName("label_2")
        self.txtPath2 = QtWidgets.QTextEdit(Dialog)
        self.txtPath2.setGeometry(QtCore.QRect(10, 110, 231, 31))
        self.txtPath2.setObjectName("txtPath2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(10, 170, 66, 19))
        self.label_3.setObjectName("label_3")
        self.txtPath3 = QtWidgets.QTextEdit(Dialog)
        self.txtPath3.setGeometry(QtCore.QRect(10, 190, 231, 31))
        self.txtPath3.setObjectName("txtPath3")
        self.btnSearchCNN = QtWidgets.QPushButton(
            Dialog, clicked=lambda: self.searchCNN())
        self.btnSearchCNN.setGeometry(QtCore.QRect(240, 40, 31, 31))
        self.btnSearchCNN.setObjectName("btnSearchCNN")
        self.btnSearchDirTest = QtWidgets.QPushButton(
            Dialog, clicked=lambda: self.searchTestDir())
        self.btnSearchDirTest.setGeometry(QtCore.QRect(240, 110, 31, 31))
        self.btnSearchDirTest.setObjectName("btnSearchDirTest")
        self.btnSearchFile = QtWidgets.QPushButton(
            Dialog, clicked=lambda: self.searchFile())
        self.btnSearchFile.setGeometry(QtCore.QRect(240, 190, 31, 31))
        self.btnSearchFile.setObjectName("btnSearchFile")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(10, 300, 271, 71))
        self.label_4.setObjectName("label_4")
        self.btnExecute = QtWidgets.QPushButton(
            Dialog, clicked=lambda: self.execute())
        self.btnExecute.setGeometry(QtCore.QRect(90, 230, 103, 36))
        self.btnExecute.setObjectName("btnExecute")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(10, 20, 131, 19))
        self.label_5.setObjectName("label_5")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate(
            "Dialog", "Convolutional neural network"))
        self.label_2.setText(_translate(
            "Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Training Directory</span></p></body></html>"))
        self.label_3.setText(_translate(
            "Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Test File</span></p></body></html>"))
        self.btnSearchCNN.setText(_translate("Dialog", "..."))
        self.btnSearchDirTest.setText(_translate("Dialog", "..."))
        self.btnSearchFile.setText(_translate("Dialog", "..."))
        self.label_4.setText(_translate(
            "Dialog", "<html><head/><body><p><br/></p></body></html>"))
        self.btnExecute.setText(_translate("Dialog", "Execute"))
        self.label_5.setText(_translate(
            "Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">CNN</span></p></body></html>"))

    def execute(self):
        classificador = Sequential()
        classificador.add(
            Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        classificador.add(BatchNormalization())
        classificador.add(MaxPooling2D(pool_size=(2, 2)))
        classificador.add(
            Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        classificador.add(BatchNormalization())
        classificador.add(MaxPooling2D(pool_size=(2, 2)))
        classificador.add(Flatten())
        classificador.add(Dense(units=128, activation='relu'))
        classificador.add(Dropout(0.2))
        classificador.add(Dense(units=128, activation='relu'))
        classificador.add(Dropout(0.2))
        classificador.add(Dense(units=1, activation='sigmoid'))
        classificador.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        gerador_treinamento = ImageDataGenerator(rescale=1/255,
                                                rotation_range=7,
                                                horizontal_flip=True,
                                                shear_range=0.2,
                                                height_shift_range=0.07,
                                                zoom_range=0.2)
        gerador_teste = ImageDataGenerator(rescale=1./255)

        base_treinamento = gerador_treinamento.flow_from_directory(dataset_training,
                                                                target_size=(
                                                                    64, 64),
                                                                batch_size=32,
                                                                class_mode='binary')

        base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

        classificador.fit_generator(base_treinamento, steps_per_epoch=1800/32,
                                    epochs=5, validation_data=base_teste,
                                    validation_steps=200/32)

        imagem_teste_hyena = image.load_img('dataset/test_set/hyena/hyena_025_val_resized.jpg',
                                            target_size=(64, 64))

        imagem_teste_hyena = image.img_to_array(imagem_teste_hyena)
        imagem_teste_hyena /= 255
        imagem_teste_hyena = np.expand_dims(imagem_teste_hyena, axis=0)

        previsao = classificador.predict(imagem_teste_hyena)
        previsao = (previsao > 0.5)

   def searchCNN(self):
        print('Button CNN pressed')

    def searchFile(self):
        filename = QFileDialog.getOpenFileName()
        test__file = filename
        self.txtPath3.setText(filename[0])

    def searchTestDir(self):
        folder = str(QFileDialog.getExistingDirectory(
            None, "Select Directory"))
        dataset_training = folder
        self.txtPath2.setText(folder)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
