from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import cv2
import sys
import os
import data_relative

# window class
class Window(QMainWindow):
    def __init__(self, nw):
        super().__init__()
        self.nw = nw
        self.setWindowTitle("Paint")

        self.setGeometry(100, 100, 280, 280) 

        self.image = QImage(self.size(), QImage.Format_RGB32) 
        self.image.fill(Qt.white)

        self.drawing = False
        self.brushSize = 10
        self.brushColor = Qt.black

        self.lastPoint = QPoint()
        mainMenu = self.menuBar()
  
        # creating menu
        fileMenu = mainMenu.addMenu("File")
        saveMenu = mainMenu.addMenu("Save")

        saveZero = QAction("0", self)
        saveMenu.addAction(saveZero)
        saveZero.triggered.connect(self.fastSave)
        saveOne = QAction("1", self)
        saveMenu.addAction(saveOne)
        saveOne.triggered.connect(self.fastSave)
        saveTwo = QAction("2", self)
        saveMenu.addAction(saveTwo)
        saveTwo.triggered.connect(self.fastSave)
        saveThree = QAction("3", self)
        saveMenu.addAction(saveThree)
        saveThree.triggered.connect(self.fastSave)
        saveFour = QAction("4", self)
        saveMenu.addAction(saveFour)
        saveFour.triggered.connect(self.fastSave)
        saveFive = QAction("5", self)
        saveMenu.addAction(saveFive)
        saveFive.triggered.connect(self.fastSave)
        saveSix = QAction("6", self)
        saveMenu.addAction(saveSix)
        saveSix.triggered.connect(self.fastSave)
        saveSeven = QAction("7", self)
        saveMenu.addAction(saveSeven)
        saveSeven.triggered.connect(self.fastSave)
        saveEight = QAction("8", self)
        saveMenu.addAction(saveEight)
        saveEight.triggered.connect(self.fastSave)
        saveNine = QAction("9", self)
        saveMenu.addAction(saveNine)
        saveNine.triggered.connect(self.fastSave)

        #Clear
        clearAction = QAction("Clear", self)
        clearAction.setShortcut("Ctrl + C")
        fileMenu.addAction(clearAction)
        clearAction.triggered.connect(self.clear)

        #Inspect
        inspectAction = QAction("Inspect", self)
        fileMenu.addAction(inspectAction)
        inspectAction.triggered.connect(self.inspect)

    def mousePressEvent(self, event): 
        if event.button() == Qt.LeftButton: 
            self.drawing = True
            self.lastPoint = event.pos() 

    def mouseMoveEvent(self, event): 
        if (event.buttons() & Qt.LeftButton) & self.drawing: 
            painter = QPainter(self.image) 
            painter.setPen(QPen(self.brushColor, self.brushSize,  
                            Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)) 
            painter.drawLine(self.lastPoint, event.pos()) 
            self.lastPoint = event.pos() 
            self.update() 

    def mouseReleaseEvent(self, event): 
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event): 
        canvasPainter = QPainter(self) 
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect()) 

    def save(self): 
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", 
                          "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ") 
        print(filePath)
        if filePath == "": 
            return
        self.image.save(filePath)
    
    def fastSave(self):
        dirname = "./data/"+self.sender().text()
        number_files = len(os.listdir(dirname))
        self.image.save(dirname+"/"+str(number_files)+".png")
        cv2.imwrite(dirname+"/"+str(number_files)+".png" ,cv2.resize(cv2.imread(dirname+"/"+str(number_files)+".png", cv2.IMREAD_UNCHANGED), (28,28)))

    def inspect(self):
        self.image.save("./data/test/inspect.png")
        cv2.imwrite("./data/test/inspect.png" ,cv2.resize(cv2.imread("./data/test/inspect.png", cv2.IMREAD_UNCHANGED), (28,28)))
        print("Result: ", np.argmax(self.nw.feedfoward(data_relative.get_inspect_data())) )

    def clear(self):
        self.image.fill(Qt.white)
        self.update()
  
App = QApplication(sys.argv)