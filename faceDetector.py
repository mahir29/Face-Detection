import cv2
import numpy as np

cap=cv2.VideoCapture(0)
detector=cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

while True:
    ret,frame=cap.read()

    if ret:
        faces=detector.detectMultiScale(frame)

        for face in faces:
            x,y,w,h=face
            
            cut=frame[y:y+h, x:x+w]

            fix = cv2.resize(cut, (100,100))
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

        cv2.imshow("My Screen",frame)
        cv2.imshow("My face", gray)

    if cv2.waitKey(5)==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
