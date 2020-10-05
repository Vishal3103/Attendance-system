from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import os
import csv
import shutil
import numpy as np
from PIL import Image
import pandas as pd
import time
import datetime


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(958, 440)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        mainWindow.setFont(font)
        self.heading1 = QtWidgets.QLabel(mainWindow)
        self.heading1.setGeometry(QtCore.QRect(18, 5, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(True)
        font.setWeight(75)
        self.heading1.setFont(font)
        self.heading1.setObjectName("heading1")
        self.IdLabel = QtWidgets.QLabel(mainWindow)
        self.IdLabel.setGeometry(QtCore.QRect(20, 60, 81, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.IdLabel.setFont(font)
        self.IdLabel.setObjectName("IdLabel")
        self.nameLabel = QtWidgets.QLabel(mainWindow)
        self.nameLabel.setGeometry(QtCore.QRect(20, 121, 71, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.nameLabel.setFont(font)
        self.nameLabel.setObjectName("nameLabel")
        self.userIDtext = QtWidgets.QTextEdit(mainWindow)
        self.userIDtext.setGeometry(QtCore.QRect(109, 64, 261, 31))
        self.userIDtext.setObjectName("userIDtext")
        self.nameText = QtWidgets.QTextEdit(mainWindow)
        self.nameText.setGeometry(QtCore.QRect(109, 128, 261, 31))
        self.nameText.setObjectName("nameText")
        self.notifLabel = QtWidgets.QLabel(mainWindow)
        self.notifLabel.setGeometry(QtCore.QRect(424, 61, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.notifLabel.setFont(font)
        self.notifLabel.setObjectName("notifLabel")
        self.notiftext = QtWidgets.QTextEdit(mainWindow)
        self.notiftext.setGeometry(QtCore.QRect(530, 65, 401, 31))
        self.notiftext.setObjectName("notiftext")
        self.heading2 = QtWidgets.QLabel(mainWindow)
        self.heading2.setGeometry(QtCore.QRect(20, 195, 281, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(True)
        font.setWeight(75)
        self.heading2.setFont(font)
        self.heading2.setObjectName("heading2")
        self.captureButton = QtWidgets.QPushButton(mainWindow)
        self.captureButton.setGeometry(QtCore.QRect(18, 260, 151, 41))
        self.captureButton.clicked.connect(self.CaptureImage)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.captureButton.setFont(font)
        self.captureButton.setObjectName("captureButton")
        self.trainButton = QtWidgets.QPushButton(mainWindow)
        self.trainButton.setGeometry(QtCore.QRect(19, 327, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.trainButton.setFont(font)
        self.trainButton.setObjectName("trainButton")
        self.trainButton.clicked.connect(self.TrainImages)
        self.progressBar = QtWidgets.QProgressBar(mainWindow)
        self.progressBar.setGeometry(QtCore.QRect(20, 390, 211, 23))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.progressBar.setFont(font)
        self.progressBar.setProperty("value", 50)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.hide()
        self.registerButton = QtWidgets.QPushButton(mainWindow)
        self.registerButton.setGeometry(QtCore.QRect(720, 261, 211, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.registerButton.setFont(font)
        self.registerButton.setObjectName("registerButton")
        self.registerButton.clicked.connect(self.RegisterAttendance)
        self.attendLabel = QtWidgets.QLabel(mainWindow)
        self.attendLabel.setGeometry(QtCore.QRect(320, 330, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.attendLabel.setFont(font)
        self.attendLabel.setObjectName("attendLabel")
        self.attendtext = QtWidgets.QTextEdit(mainWindow)
        self.attendtext.setGeometry(QtCore.QRect(530, 335, 401, 31))
        self.attendtext.setObjectName("attendtext")

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "FRAS"))
        self.heading1.setText(_translate("mainWindow", "User Details"))
        self.IdLabel.setText(_translate("mainWindow", "User ID"))
        self.nameLabel.setText(_translate("mainWindow", "Name"))
        self.notifLabel.setText(_translate("mainWindow", "Notification"))
        self.heading2.setText(_translate("mainWindow", "Register Attendance"))
        self.captureButton.setText(_translate("mainWindow", "Capture Image"))
        self.trainButton.setText(_translate("mainWindow", "Train Image"))
        self.registerButton.setText(_translate("mainWindow", "Register Attendance"))
        self.attendLabel.setText(_translate("mainWindow", "Attendance notification"))

    def CaptureImage(self):
        Id = self.userIDtext.toPlainText()
        name = self.nameText.toPlainText()

        if(name.isalpha()):
            cam = cv2.VideoCapture(0)
            harcascadePath = "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(harcascadePath)
            sampleNum = 0
            while(True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    sampleNum = sampleNum+1
                    cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                    cv2.imshow('frame',img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                elif sampleNum>59:
                    break
            cam.release()
            cv2.destroyAllWindows()
            res = "Images saved for ID: " + Id + "," + " Name : "+ name
            row = [Id , name]
            with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
            self.notiftext.setPlainText(res)
        else:
            msg1 = "The user ID and name fields are empty or invalid"
            self.notiftext.setPlainText(msg1)

    def TrainImages(self):
        self.progressBar.show()
        self.progressBar.setProperty("value", 0)
        recognizer = cv2.face_LBPHFaceRecognizer.create()
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        faces,Id = self.getImagesAndLabels(".\TrainingImage")
        recognizer.train(faces, np.array(Id))
        recognizer.save("FeatureFiles\TrainingImage.yml")
        res = "Model training success!"
        self.notiftext.setPlainText(res)
        self.progressBar.setProperty("value", 100)

    def getImagesAndLabels(self,path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        faces = []
        Ids = []
        for imagePath in imagePaths:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage,'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(Id)
        return faces,Ids

    def RegisterAttendance(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("FeatureFiles\TrainingImage.yml")
        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath)
        df = pd.read_csv("StudentDetails\StudentDetails.csv")
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names =  ['Id','Name','Date','Time']
        attendance = pd.DataFrame(columns = col_names)
        while True:
            ret, im = cam.read()
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2,5)
            for(x,y,w,h) in faces:
                cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
                Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
                if(conf < 50):
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    aa = df.loc[df['Id'] == Id]['Name'].values
                    tt = str(Id)+"-"+aa
                    attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                else:
                    Id = 'Unknown'
                    tt = str(Id)
                if(conf > 75):
                    noOfFile = len(os.listdir("ImagesUnknown"))+1
                    cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])
                cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)
            attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
            cv2.imshow('User',im)
            if (cv2.waitKey(1)==ord('q')):
                break
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour,Minute,Second = timeStamp.split(":")
        fileName ="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
        attendance.to_csv(fileName,index=False)
        cam.release()
        cv2.destroyAllWindows()
        res = "Attendance Log Success"
        self.attendtext.setPlainText(res)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QDialog()
    ui = Ui_mainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
