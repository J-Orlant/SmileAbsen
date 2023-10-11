import cv2
import os
from simple_facerec import SimpleFacerec
from datetime import datetime

# Endode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("./image")

# Load and Set all assets data
folderModePath = './Design/Card'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
backgroundBase = cv2.imread("./Design/MainCard.png")
backgroundBase[44:44 + 633, 808:808 + 414] = imgModeList[0]
# print(imgModeList[0])

# Smile classifier
smile_detector = cv2.CascadeClassifier("./haarcascade/haarcascade_smile.xml")
# Load Camera
cap = cv2.VideoCapture(0)
cap.set(3, 650)
cap.set(4, 500)

while True:
    ret, frame = cap.read()

    # Detect Face
    face_locations, face_name = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_name):
        name = name.split("-")
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        
        # Detect Senyum mu
        smiles = smile_detector.detectMultiScale(frame, scaleFactor=2, minNeighbors=20)
        if len(smiles) > 0:
            backgroundBase[44:44 + 633, 808:808 + 414] = cv2.resize(imgModeList[1], (414, 633))
            # Get current date and time
            now = datetime.now()
            now = now.strftime("%d/%m/%Y %H:%M:%S")
            
            # Set information text
            cv2.putText(backgroundBase, str(name[0]), (920, 385), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
            cv2.putText(backgroundBase, str(name[1]), (965, 455), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(backgroundBase, now, (925, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            
            # Set image user
            userPhoto = cv2.imread(os.path.join('./image/', name[0] + '-' + name[1] + '.jpg'))
            userPhoto = cv2.resize(userPhoto, (260, 255), fx=0.12, fy=0.12)
            backgroundBase[90:90+255, 885:885+260] = userPhoto


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        # bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
        # backgroundBase = cvzone.cornerRect(backgroundBase, bbox, rt=0)
    
    frame = cv2.resize(frame, (650, 500))
    backgroundBase[157:157+500, 50:50+650] = frame
    
    cv2.imshow("Smile Absen", backgroundBase)
    
    key = cv2.waitKey(1)
    
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()