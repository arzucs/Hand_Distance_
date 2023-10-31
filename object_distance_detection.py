import cv2
import math
import cvzone
import imutils
import numpy as np
import matplotlib as plt
import cvzone
from cvzone.HandTrackingModule import HandDetector

cap=cv2.VideoCapture(0)

detector= HandDetector(detectionCon= 0.6) #güvenilirlik değeri(detectionCon) ne düşükse eli o kadar iyi bulur

#find the function
# x mesafe y cm cinsinden değeri
x = [650,450,400,300,245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [ 5,10,14,20,25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

coff=np.polyfit(x,y,2) # y = Ax^2+ Bx+ C

while True:
    ret, frame=cap.read()
    frame=cv2.flip(frame, 1)
    frame=imutils.resize(frame, width=1280,height=720)
    hands, frame = detector.findHands(frame, flipType=False)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        x, y, w, h=hands[0]["bbox"] 
        
        x1 , y1= lmList[5][0] , lmList[5][1] #[4, 278, -91] 4=baş parmak ,278=0.eleman, -91= 1.eleman 
        x2 , y2 =lmList[17][0] , lmList[17][1]
         
        distance= int(math.sqrt((y2 -y1 )**2 + (x2 -x1 )**2 ))
        distance = int(math.hypot(x2-x1 ,y2-y1 )) # 4 ve 8 arası çizgi uzunluğu hesaplar
        A, B, C = coff # yukardaki 2.dereceden polinom olan y = Ax^2+ Bx+ C deki katsayılara eşit
        distanceCM = (A*(distance**2) + (B * distance )+ C) #santimetre cinsinden mesafe miktarı
        print(distanceCM,distance)
        # print(distance)
        # print("El Noktaları:", lmList)
        
        cvzone.putTextRect(frame, f'{int (distanceCM )} cm',(x + 100, y - 30 ),2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    
cv2.destroyAllWindows()