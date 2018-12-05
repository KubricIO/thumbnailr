import cv2
import os
import requests
import urllib2


video_url = 'https://storage.googleapis.com/videos-kubric/video-c67d4b7f-8ba7-40b6-8269-fe09cb33c18b.mp4'
file_name = 'trial_video.mp4' 
rsp = requests.urlopen(video_url)
with open(file_name,'wb') as f:
    f.write(rsp.read())

# we can also just read in the video without actually downloading it

file1 = file_name
vidcap = cv2.VideoCapture(file1)
success,image = vidcap.read()

print (success)
count = 0
time = 0
path = "./frames"          ##making the new directory
if not os.path.isdir(path):
    os.makedirs(path)
success = True
while success:
    vidcap.set(cv2.CAP_PROP_POS_MSEC,time)
    cv2.imwrite(os.path.join(path,"frame1%d.jpg" %count), image)     
    #save frame as JPEG file
    success,image = vidcap.read()
    #print ('Read a new frame: ', success)
    count += 1
    time += 500
