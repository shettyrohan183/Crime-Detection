import cv2
import os
import face_recognition
import datetime as dt
def captureFrame():
    cap = cv2.VideoCapture("./vid.mp4")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count == 10:
            break
        locations = face_recognition.face_locations(frame)
        if len(locations) > 0:
            for i in locations:
                cropImg = frame[i[0]:i[2],i[3]:i[1]]
                cv2.imwrite(os.path.join(os.getcwd()+'/1frameImages',str(dt.datetime.now()).replace(":","_")+".jpg"),cropImg)
    
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break






def deleteDuplicates():
    dup = [] 
    for i in os.listdir("savedImages"):
        if face_recognition.face_encodings(cv2.imread("savedImages/"+i)) == []:
            os.remove("./savedImages/"+i)
    imglist = os.listdir("savedImages")
    for i in range(len(imglist)):
        if imglist[i] not in dup:
            for j in range(i+1,len(imglist)):
                if compareFaces(imglist[i],imglist[j]):
                    if imglist[j] not in dup:
                        dup.append(imglist[j])
    
    for i in dup:
        os.remove("./savedImages/"+i)




