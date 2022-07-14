#!/usr/bin/env python
# coding: utf-8

# In[8]:


from tkinter import *
import os
import cv2
import numpy as np
from PIL import Image,ImageTk
from keras.models import load_model


from pygrabber.dshow_graph import FilterGraph
import face_recognition

import time
import datetime as dt
from collections import deque


# In[9]:


# cap = cv2.VideoCapture(0)


# In[10]:


import os
import glob
from os import listdir
string = "local"
if string=="local":
    files = glob.glob(r'E:\go_ai\last-crime\data\sam\vi\*')
else:
    files = glob.glob(r'E:\go_ai\last-crime\data\our\vid\*')
for f in files:
    os.remove(f)


# In[23]:


for i  in os.listdir(r"E:\go_ai\last-crime\video"):
    SEQUENCE_LENGTH = 20
    name = r"E:\go_ai\last-crime\video" + "\\"
    IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224
    cap = cv2.VideoCapture(name +i)
    print(i)
#     cap = 'E:\go_ai\last-crime\video\12-8831_77-6164.avi'
    print(cap)
    
    frames_queue = deque(maxlen = 20)
    t1 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
#         print("this while")
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
#         cv2.imwrite(os.path.join(os.getcwd()+'/savedImages',str(dt.datetime.now()).replace(":","_")+".jpg"),normalized_frame)
        frames_queue.append(normalized_frame)
        
#     if len(frames_queue) == SEQUENCE_LENGTH:
# #         predictions = model.predict(np.expand_dims(frames_queue, axis = 0))[0]
#         fight.config(text ="Abnormal: "+ str(predictions[0]))
#         NonFight.config(text ="Normal: "+ str(predictions[1]))
        
        m = face_recognition.face_locations(frame)
        print(len(m))
        if len(m) > 0 :
            for i in range(len(m)):
                cv2.putText(frame, "Face detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.rectangle(frame,(m[i][3],m[i][0]),(m[i][1],m[i][2]),(0,255,0),2)
                if (int(time.time()) - int(t1)) % 1 == 0:
                    cv2.putText(frame, "Saving", (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    # cv2.rectangle(frame,(m[i][3],m[i][0]),(m[i][1],m[i][2]),(255,0,0),2)
                    print("saving")
                    cv2.imwrite(os.path.join(os.getcwd()+'/savedImages',str(dt.datetime.now()).replace(":","_")+".jpg"),frame)
                    if m==1:
                        break
                # temp = frame[m[i][0]:m[i][2],m[i][3]:m[i][1]]     
                    t1 = time.time()


# In[12]:


for i  in os.listdir(r"E:\go_ai\last-crime\video"):
    print(i)


# In[16]:


import cv2
import os
import face_recognition
import datetime as dt
def captureFrame():
    cap = cv2.VideoCapture(r"E:\go_ai\last-crime\video\12-8831_77-6164.avi")
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
            cap.release()
            break
        


# In[6]:


captureFrame()


# In[ ]:




