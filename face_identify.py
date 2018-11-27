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
for f in flist:
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
            pass
        else:
            os.remove(path+"/"+f)
#print("done")

def ploting():
    # PLOT
    plt.imshow(img_gray, cmap=plt.get_cmap("gray"))
    ca = plt.gca()
    for face in faces:
        ca.add_patch(Rectangle((face[0], face[1]), face[2], face[3]
                               , fill=None, alpha=1, edgecolor='red'))
    plt.title("Face detection with Viola-Jones")
    plt.show()
