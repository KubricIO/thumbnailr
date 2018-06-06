import urllib.request
from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv


import json
import numpy as np 

import shutil
from requests import get

# from config import api_key
import os


api_key = 'AIzaSyAfZf9vtUshHH6evBh20M8MV6WhI57fT6k'
videoId=[]
name = []

# function for downloading thumbnails
def download(url, file_name):
        # open in binary mode
        with open(file_name, "wb") as file:
            # get request
            response = get(url)
            # write to file
            file.write(response.content)

list1= []
list2=[]

for r in range(10000):
	name.append(r)

name1 = [str(item) for item in name]  
# print("getting video ids:")



for i in range(10000):
	source = requests.get('https://www.incognitube.com/').text
	soup = BeautifulSoup(source,'lxml')
	all_links = soup.find_all('div',{'id':'info'})

	row=[]

	for link in all_links:
		row.append(link.find_all('a'))
	# print(row,'\n')

	row2=[]

	for link in row[0]:
		row2.append(link.get('href'))

	# print(row2)	

	string = str(row2[1])

	segments = string.rpartition('v=')
	segments2 = segments[2].rpartition('&')

	videoId.append(segments2[0])

	print("i=",i)
	val= videoId[i]
	if any(val in string for string in list1): # for skipping duplicate images
		print('skipped')
		continue
	else:
		url = 'https://img.youtube.com/vi/'+videoId[i]+'/hqdefault.jpg'
		filename = './'+name1[i]+'b'+'.jpg'
		# urllib.request.urlretrieve(url, r)
		download(url,filename)
		list1.append(videoId[i])
		# i+=1
# print("downloading thumbnails")
  
# print(name1)

# for r in range(len(videoId)):
#     url = 'https://img.youtube.com/vi/'+videoId[r]+'/hqdefault.jpg'
#     # print(url)
#     filename = './'+name1[r]+'b'+'.jpg'
#     # urllib.request.urlretrieve(url, r)
#     download(url,filename)
#     print("r=",r)
# # print(allChannelIds[i])

