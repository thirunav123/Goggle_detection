import sys,os,cv2,threading,pathlib
import numpy as np
import tensorflow as tf


from PyQt5.uic import loadUi
from PyQt5 import QtWidgets,QtGui,QtCore
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QDesktopWidget
from PyQt5.QtCore import Qt, QPoint
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from matplotlib import pyplot as plt

current_path=pathlib.Path(__file__).parent.resolve()
print(current_path)

folders_path={
    'DETECTION_RECORD': os.path.join(current_path,'Detection_records'),
    'ML':os.path.join(current_path,'ML_files'),
    'MODEL':os.path.join(current_path,'ML_files','model')

}
files_path={
    'Ui1': os.path.join(current_path,'page1.ui'),
    'Ui2':os.path.join(current_path,'page2.ui'),
    'Admin_file': os.path.join(current_path,'admin.pbtxt'),
    'LABELMAP': os.path.join(current_path, folders_path['ML'],'label_map.pbtxt'),
    'PIPELINE_CONFIG': os.path.join(current_path, folders_path['ML'],'model','pipeline.config')
}

admin_file = open(files_path['Admin_file'], "r")
content = admin_file.read()
admins = content.split(",")
admin_file.close()
print(admins)

liverun_flag = False
label_flag = False
shown_flag = False
current_posX=0
current_posY=0
x=200
y=1100
w=640
h=640
x1,x2,y1,y2=100,100,100,100

if not os.path.exists(folders_path['DETECTION_RECORD']):
    os.mkdir(folders_path['DETECTION_RECORD'])

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files_path['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(folders_path['MODEL'], 'ckpt-6')).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(files_path['LABELMAP'])


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    #print(shapes)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
    
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
        global current_posX,current_posY,shown_flag,movex_limit,movey_limit
        super(firstWindow, self).__init__()
        loadUi(files_path['Ui1'], self)
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
        self.limit_x=movex_limit-self.geometry().width()
        self.limit_y=movey_limit-self.geometry().height()

    def next(self):
        gen = self.lineEdit_3.text()
        print(bool(len(gen)))
        if bool(len(gen)):
            print(gen)
            if gen.isdigit():
                if gen in admins:
                    self.second=secondWindow(gen)
                    self.close()
                else:
                    mbox = QMessageBox()  # popup the message box widget
                    mbox.setWindowTitle("Warning")
                    mbox.setText("You are not admin")
                    mbox.setIcon(QMessageBox.Warning)
                    x = mbox.exec_()
            else:
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
        # print(self.x(),self.y())

    def mouseMoveEvent(self, event):
        global current_posX,current_posY
        delta= QPoint(event.globalPos() - self.oldPosition)
        # print(x,y,self.x(),self.y(),current_posX,current_posY)
        current_posX=self.x()+delta.x()
        current_posY=self.y()+delta.y()

        if (0 <= current_posX <= self.limit_x):
            #self.move(current_posX,current_posY)
            if (0 <= current_posY <= self.limit_y):
                self.move(current_posX,current_posY)
            elif (current_posY<0):
                self.move(current_posX,0)
            elif(current_posY>self.limit_y):
                self.move(current_posX,self.limit_y)
        elif (current_posX<0):
            #self.move(0,current_posY)
            if (0 <= current_posY <= self.limit_y):
                self.move(0,current_posY)
            elif (current_posY<0):
                self.move(0,0)
            elif(current_posY>self.limit_y):
                self.move(0,self.limit_y)
        elif(current_posX>self.limit_x):
            #self.move(self.limit_x,current_posY)
            if (0 <= current_posY <= self.limit_y):
                self.move(self.limit_x,current_posY)
            elif (current_posY<0):
                self.move(self.limit_x,0)
            elif(current_posY>self.limit_y):
                self.move(self.limit_x,self.limit_y)
        self.oldPosition=event.globalPos()

class secondWindow(QMainWindow):
    def __init__(self, genid):
        global current_posX,current_posY,movex_limit,movey_limit
        super(secondWindow, self).__init__()
        loadUi(files_path['Ui2'], self)
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
        self.limit_x=movex_limit-self.geometry().width()
        self.limit_y=movey_limit-self.geometry().height()


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
        os.system('start {}'.format(folders_path['DETECTION_RECORD']))

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

    def mousePressEvent(self,event):
        # print(event.globalPos())
        self.oldPosition=event.globalPos()

    def mouseMoveEvent(self, event):
        global current_posX,current_posY
        delta= QPoint(event.globalPos() - self.oldPosition)
        # print(x,y,self.x(),self.y(),current_posX,current_posY)
        current_posX=self.x()+delta.x()
        current_posY=self.y()+delta.y()

        if (0 <= current_posX <= self.limit_x):
            #self.move(current_posX,current_posY)
            if (0 <= current_posY <= self.limit_y):
                self.move(current_posX,current_posY)
            elif (current_posY<0):
                self.move(current_posX,0)
            elif(current_posY>self.limit_y):
                self.move(current_posX,self.limit_y)
        elif (current_posX<0):
            #self.move(0,current_posY)
            if (0 <= current_posY <= self.limit_y):
                self.move(0,current_posY)
            elif (current_posY<0):
                self.move(0,0)
            elif(current_posY>self.limit_y):
                self.move(0,self.limit_y)
        elif(current_posX>self.limit_x):
            #self.move(self.limit_x,current_posY)
            if (0 <= current_posY <= self.limit_y):
                self.move(self.limit_x,current_posY)
            elif (current_posY<0):
                self.move(self.limit_x,0)
            elif(current_posY>self.limit_y):
                self.move(self.limit_x,self.limit_y)
        self.oldPosition=event.globalPos()



app = QApplication(sys.argv)
movex_limit=app.desktop().screenGeometry().width()
movey_limit=app.desktop().screenGeometry().height()
# print(QDesktopWidget().availableGeometry().width())
ex = firstWindow()
sys.exit(app.exec_())
