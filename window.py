

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QFileInfo
from PyQt5.QtWidgets import QFileDialog
from keras import models
from keras.preprocessing import image
import numpy as np


class Ui_Dialog(object):

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(295, 301)
        self.txtPath1 = QtWidgets.QTextEdit(Dialog)
        self.txtPath1.setGeometry(QtCore.QRect(10, 40, 231, 31))
        self.txtPath1.setObjectName("txtPath1")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(10, 80, 66, 19))
        self.label_3.setObjectName("label_3")
        self.txtPath3 = QtWidgets.QTextEdit(Dialog)
        self.txtPath3.setGeometry(QtCore.QRect(10, 100, 231, 31))
        self.txtPath3.setObjectName("txtPath3")
        self.btnSearchCNN = QtWidgets.QPushButton(
            Dialog,  clicked=lambda: self.searchModel())
        self.btnSearchCNN.setGeometry(QtCore.QRect(240, 40, 31, 31))
        self.btnSearchCNN.setObjectName("btnSearchCNN")
        self.btnSearchFile = QtWidgets.QPushButton(
            Dialog,  clicked=lambda: self.searchTestFile())
        self.btnSearchFile.setGeometry(QtCore.QRect(240, 100, 31, 31))
        self.btnSearchFile.setObjectName("btnSearchFile")
        self.btnExecute = QtWidgets.QPushButton(
            Dialog,  clicked=lambda: self.execute())
        self.btnExecute.setGeometry(QtCore.QRect(170, 200, 103, 36))
        self.btnExecute.setObjectName("btnExecute")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(10, 20, 131, 19))
        self.label_5.setObjectName("label_5")
        self.txtResult = QtWidgets.QLabel(Dialog)
        self.txtResult.setGeometry(QtCore.QRect(10, 250, 261, 41))
        self.txtResult.setText("")
        self.txtResult.setObjectName("txtResult")
        self.txtPath3_2 = QtWidgets.QTextEdit(Dialog)
        self.txtPath3_2.setGeometry(QtCore.QRect(10, 160, 231, 31))
        self.txtPath3_2.setObjectName("txtPath3_2")
        self.btnSearchFile_2 = QtWidgets.QPushButton(
            Dialog,  clicked=lambda: self.searchWeights())
        self.btnSearchFile_2.setGeometry(QtCore.QRect(240, 160, 31, 31))
        self.btnSearchFile_2.setObjectName("btnSearchFile_2")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(10, 140, 111, 19))
        self.label_4.setObjectName("label_4")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate(
            "Dialog", "Convolutional neural network"))
        self.label_3.setText(_translate(
            "Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Test File</span></p></body></html>"))
        self.btnSearchCNN.setText(_translate("Dialog", "..."))
        self.btnSearchFile.setText(_translate("Dialog", "..."))
        self.btnExecute.setText(_translate("Dialog", "Execute"))
        self.label_5.setText(_translate(
            "Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Model</span></p></body></html>"))
        self.btnSearchFile_2.setText(_translate("Dialog", "..."))
        self.label_4.setText(_translate(
            "Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Select Weights</span></p></body></html>"))

    def execute(self):
        print('Executou')
        self.model = models.load_model(self.dir_path)

        if (self.weights_file):
            self.model.load_weights(str(self.weights_file))

        imageToPredict = image.load_img(self.image, target_size=(64, 64))
        imageToPredict = image.img_to_array(imageToPredict)
        imageToPredict /= 255
        imageToPredict = np.expand_dims(imageToPredict, axis=0)

        self.prediction = self.model.predict(imageToPredict)
        print(self.prediction)
        if(self.prediction > 0.5):
            self.txtResult.setText("It's hyena")
        else:
            self.txtResult.setText("It's cheetah")

    def searchTestFile(self):
        filename = QFileDialog.getOpenFileName()
        self.image = filename[0]
        self.txtPath3.setText(self.image)

    def searchWeights(self):
        weights_folder = QFileDialog.getOpenFileName()
        self.weights_file = weights_folder[0]
        self.txtPath3_2.setText(self.weights_file)

    def searchModel(self):
        self.dir_path = str(QFileDialog.getExistingDirectory(
            None, "Select Directory"))
        self.txtPath1.setText(self.dir_path)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
