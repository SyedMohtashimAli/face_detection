#face detection is performed using classifier
import cv2
import numpy as np
#using HAAR a type of classifier

face_casscade = cv2.CascadeClassifier('casscade/haarcascade_frontalface_default.xml')
eye_casscade = cv2.CascadeClassifier('casscade/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_casscade.detectMultiScale(gray,1.3,5)
    #To draw a rectangle on faces  ( img  ,  starting_point  ,  ending_point , color_of_rectangle(B-R-G format) , width_of_the_line)
    for(x,y,h,w) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #ROI => REGION OF INTEREST
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        #if face has been detected then detect eyes on that face in ROI
        eyes = eye_casscade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img',img)
    k=cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
