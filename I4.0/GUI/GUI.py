import sys,os,cv2,threading,pathlib
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets,QtGui,QtCore
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QDesktopWidget

from PyQt5.QtCore import Qt, QPoint

current_path=pathlib.Path(__file__).parent.resolve()
print(current_path)
admin = [123, 15000, 15001]
# oldPosition=QPoint()
liverun_flag = False
label_flag = False
shown_flag = False
paths=os.path.join(current_path,'Detection_records')
Ui1_path=os.path.join(current_path,'page1.ui')
Ui2_path=os.path.join(current_path,'page2.ui')
current_posX=0
current_posY=0

if not os.path.exists(paths):
    os.mkdir(paths)

def live():
    global liverun_flag,label_flag
    cap = cv2.VideoCapture(r'D:\Videos_SPR4.0\2021-11-24\1.mp4')
    while True:
        ret, or_read_img = cap.read()
        # keyCode = cv2.waitKey(1)
        cv2.imshow("Live_detection", or_read_img)
        # if (cv2.waitKey(1) & 0xFF == ord('q')) or flag == 1:
        key = cv2.waitKey(20)
        if key == 27 or liverun_flag == False:
            cap.release()
            cv2.destroyAllWindows()
            break


class firstWindow(QtWidgets.QMainWindow):
    def __init__(self):
        global current_posX,current_posY,shown_flag
        super(firstWindow, self).__init__()
        loadUi(Ui1_path, self)
        self.setWindowFlags(Qt.FramelessWindowHint)
        print("Ui_1")
        self.Indicator_label.setStyleSheet("border: 5px solid black; background-color: rgb(0, 255, 0);border-radius: 20px;")
        self.continueButton.clicked.connect(self.next)
        self.exitButton.clicked.connect(self.exit)
        if not shown_flag:
            self.show()
            current_posX=self.pos().x()
            current_posY=self.pos().y()
            shown_flag = True
        else:
            self.move(current_posX,current_posY)
            self.show()

    def next(self):
        gen = self.lineEdit_3.text()
        print(bool(len(gen)))
        if bool(len(gen)):
            try:
                if int(gen) in admin:
                    self.second=secondWindow(gen)
                    self.close()
                else:
                    mbox = QMessageBox()  # popup the message box widget
                    mbox.setWindowTitle("Warning")
                    mbox.setText("You are not admin")
                    mbox.setIcon(QMessageBox.Warning)
                    x = mbox.exec_()
            except:
                mbox = QMessageBox()  # popup the message box widget
                mbox.setWindowTitle("Warning")
                mbox.setText("Gen ID must a number")
                mbox.setIcon(QMessageBox.Warning)
                x = mbox.exec_()

        else:
            mbox = QMessageBox()  # popup the message box widget
            mbox.setWindowTitle("Warning")
            mbox.setText("oops...  \nEnter a valid ID")
            mbox.setIcon(QMessageBox.Warning)
            x = mbox.exec_()

    def exit(self):
        app.quit()

    def mousePressEvent(self,event):
        self.oldPosition=event.globalPos()

    def mouseMoveEvent(self, event):
        global current_posX,current_posY
        delta= QPoint(event.globalPos() - self.oldPosition)
        current_posX=self.x()+delta.x()
        current_posY=self.y()+delta.y()
        self.move(current_posX,current_posY)
        self.oldPosition=event.globalPos()

class secondWindow(QMainWindow):
    def __init__(self, genid):
        global current_posX,current_posY
        super(secondWindow, self).__init__()
        loadUi(Ui2_path, self)
        self.setWindowFlags(Qt.FramelessWindowHint)
        print("Ui_2")
        self.t1 = threading.Thread(target=live)
        self.liveButton.clicked.connect(self.startthread)
        self.detectionRecordsButton.clicked.connect(self.showrecords)
        self.logoutButton.clicked.connect(self.logout)
        self.exitButton.clicked.connect(self.exit)
        self.genidLabel.setText(str(genid))
        self.move(current_posX,current_posY)
        self.show()

    def startthread(self):
        global liverun_flag
        if not (self.t1.is_alive()):
            self.t1 = threading.Thread(target=live)
            liverun_flag = True
            self.t1.start()
            #self.statusLabel.setText("Live is running")
            #self.statusLabel.show()
        else:
            mbox = QMessageBox()  # popup the message box widget
            mbox.setWindowTitle("Warning")
            mbox.setText("Already running")
            mbox.setIcon(QMessageBox.Warning)
            x = mbox.exec_()
            print("Already running")

    def showrecords(self):
        os.system('start {}'.format(paths))

    def logout(self):
        global liverun_flag
        liverun_flag = False
        print("Logged out")
        if (self.t1.is_alive()):
            self.t1.join()
        self.home = firstWindow()
        self.close()
        liverun_flag = 0

    def exit(self):
        global liverun_flag
        liverun_flag = False
        if (self.t1.is_alive()):
            self.t1.join()
        app.quit()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def mousePressEvent(self,event):
        print(event.globalPos())
        self.oldPosition=event.globalPos()

    def mouseMoveEvent(self, event):
        global current_posX,current_posY
        delta= QPoint(event.globalPos() - self.oldPosition)
        current_posX=self.x()+delta.x()
        current_posY=self.y()+delta.y()
        self.move(current_posX,current_posY)
        self.oldPosition=event.globalPos()
    
    


app = QApplication(sys.argv)
ex = firstWindow()
sys.exit(app.exec_())
