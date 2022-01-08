
import face_recognition
import numpy as np
import cv2
import face_recognition
import os

path = 'images'
images = []
classnames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}\{cl}')
    images.append(curImg)                                       # this will call image by its name
    classnames.append(os.path.splitext(cl)[0])
print(classnames)


def findencodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListknown = findencodings(images)
print('encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB) 

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)


    for encodeface,faceloc in zip(encodeCurFrame,facesCurFrame):
        match = face_recognition.compare_faces(encodeListknown,encodeface)
        facedist = face_recognition.face_distance(encodeListknown,encodeface)
        print(facedist)
        matchIndex = np.argmin(facedist)        #this will give no. from 0 e.g. 0  1  2

        if match[matchIndex]:
            name = classnames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    
    cv2.imshow("webcam",img)
    cv2.waitKey(1)