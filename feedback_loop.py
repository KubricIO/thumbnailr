import cv2
import os
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


path = '/home/akhil/thumbnailr/overall'   # change the path to directory
files = os.listdir(path)
count = get_filecount(path)

def feedback_loop():


    for i in range(count):
        print ("i = ",i)
        img = cv2.imread(str(i+1)+'.jpg', 0)
        cv2.imshow("image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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




