import cv2
import os
from PIL import Image
import subprocess

import numpy as np
import requests
#import urllib2

import shutil
from uuid import uuid4

def get_filecount(path_to_directory):
    if os.path.exists(path_to_directory):
        path,dirs,files = os.walk(path_to_directory).__next__()
        file_count = len(files)
        return file_count
    else :
        print("path does not exist")
        return 0


path = '/home/karthik/thumbnailr/overall/'   # change the path to directory
if not os.path.isdir(path):
    os.makedirs(path)

files = os.listdir(path)
count = get_filecount(path)
print(count)

def feedback_loop():

    for i in range(2):
        print ("i = ",i)
        #p = subprocess.Popen(path + str(i + 1) + '.jpg')
        #folder = input()
        #p.kill()
        image = Image.open(path + str(i + 1) + '.jpg',"r")
        image.show()
        folder = input()
        #os.kill()
        image.close()
        #cv2.waitKey(1000)
        #cv2.destroyAllWindows()
        i +=1



    # path1 = './' + 'folder' + str(score)
    # create_directory(path1)
    # path2 = './' + 'folder' + str(score)
    # create_directory(path2)
    # path3 = './' + 'folder' + str(score)
    # create_directory(path3)
    # path4 = './' + 'folder' + str(score)
    # create_directory(path4)
    # path5 = './' + 'folder' + str(score)
    # create_directory(path5)
    #



feedback_loop()




