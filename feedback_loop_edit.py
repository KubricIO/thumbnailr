import cv2
import os
from PIL import Image
import subprocess
import numpy as np
import requests
#import urllib2

import shutil
from uuid import uuid4

def create_directory(directory):
    """
        :param directory: folder path which needs to be created if not already present
        :return: nothing
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

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

path1 = './' + 'folder1'
create_directory(path1)
path2 = './' + 'folder2'
create_directory(path2)
path3 = './' + 'folder3'
create_directory(path3)
path4 = './' + 'folder4'
create_directory(path4)
path5 = './' + 'folder5'
create_directory(path5)
path6 = './' + 'folder6'
create_directory(path6)


def feedback_loop():

    for i in range(5):
        print ("i = ",i)
        image = Image.open(path + str(i + 1) + '.jpg',"r")
        image.show()
        folder = input()
        shutil.copy2(path + str(i + 1) + '.jpg','./'+ 'folder'+folder)
        os.popen('killall display')
        image.close()
        i +=1




feedback_loop()




