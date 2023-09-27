from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow
import sys


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        width = 800
        height = 800
        self.setGeometry(200, 200, width, height)
        self.setWindowTitle("First App")
        self.initUI()

    def initUI(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setText("Hello")
        self.label.move(50,50)

        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText("Click me")
        self.b1.clicked.connect(self.clicked)

    def clicked(self):
        self.label.setText("Clicked the button!")
        self.update()
    
    def update(self):
        self.label.adjustSize()


def buttonPress():
    print("test")


def window():
    app = QApplication(sys.argv)
    win = MyWindow()

    win.show()
    sys.exit(app.exec())

    
window()
