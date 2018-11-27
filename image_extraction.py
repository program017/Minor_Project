import cv2
import time
import os

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
      vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000*6*2*2))    
      success,image = vidcap.read()
      #print ('Read a new frame: ', success)
      cv2.imwrite( pathOut + "\\frame%d.jpg" % count, image)    
      count = count + 1
      print("Extracted frame no: ",count)
    os.remove(pathOut+"frame"+str(count-1)+".jpg")

cwd=os.getcwd()
path=cwd+"/input/"
flist=os.listdir(path)

valid_exts=[".mkv",".3gp",".mp4",".m4a",".m4v",".f4v",".f4a",".m4b",".m4r",".f5b",".mov",".3gp2",".3g2",".3gpp",".3gpp2",".ogg",".oga",".ogv",".ogx",".webm",".flv",".hdv"]
for f in flist:
    if os.path.splitext(f)[1].lower() not in valid_exts:
        print("Wrong file format. Will try again")
    else:
        input_loc=path+f
        output_loc=cwd+"/images/"
        extractImages(input_loc, output_loc)

