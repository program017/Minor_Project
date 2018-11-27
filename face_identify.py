import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  

cwd = os.getcwd()
clsf_path = cwd + "/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(clsf_path)

path = cwd + "/images"
flist = os.listdir(path)
valid_exts = [".jpg",".gif",".png",".tga", ".jpeg"]
count1=0
count2=0
for f in flist:
    count1+=1
    print(f,count1)
    if os.path.splitext(f)[1].lower() not in valid_exts:
        os.remove(path+"/"+f)
    else:
        fullpath = os.path.join(path, f)
        img_bgr = cv2.imread(fullpath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(img_gray)
        #print(f+" "+str(len(faces)))
        if len(faces)==1:
            count2+=1
            pass
        else:
            os.remove(path+"/"+f)


print("Kept: ",count2," Deleted: ",count1-count2)
