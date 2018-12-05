import requests
import shutil

video_url = 'https://www.sample-videos.com/video/mp4/240/big_buck_bunny_240p_30mb.mp4'

def download_file(url):
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

    return local_filename

import cv2
import math

filename=download_file(video_url)

videoFile = filename #Ex: "big_buck_bunny_240p_30mb.mp4"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename = "./image_" +  str(int(frameId)) + ".jpg"
        cv2.imwrite(filename, frame)
        print(filename)
cap.release()
print("Done!")
