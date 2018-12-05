import requests
# import urllib3.requests
from bs4 import BeautifulSoup
# import urllib.request
import json
import numpy as np 
# import urllib.request
import shutil
from requests import get
import pandas as pd
import csv
# from config import api_key
import os
import time

# api_key = 


for r in range(22,100):
	source = requests.get('https://www.themoviedb.org/movie?page='+str(r+1)).text
	soup = BeautifulSoup(source,'lxml')

	all_links = soup.find_all('div',{'class':'image_content'})
	# all_links = soup.find_all('div', {'style':'float: left; width: 350px; line-height: 25px;'})
	row = []

	for link in all_links:
		row.append(link.find('img'))

	row2=[]

	for link in row:
		row2.append(link.get('data-src'))

	def download(url, file_name):
	        # open in binary mode
	        with open(file_name, "wb") as file:
	            # get request
	            response = get(url)
	            # write to file
	            file.write(response.content)

	for i,link in enumerate(row2):
		filename=str(r)+'_'+str(i)+'.jpg'
		download(link,filename)            
	if r%5 == 0 :
		time.sleep(5)
	print('r=',r)
	
