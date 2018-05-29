import urllib.request
import json
import numpy as np 
import urllib.request
import shutil
from requests import get
import pandas as pd
import csv
import os

# username = 'marquesbrownlee'

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' +  directory)

# Example createFolder('./data/')
# Creates a folder in the current directory called data

API = open("api_key.txt", 'r')
if API.mode == 'r':
    api_key = API.read()

# Get an api key and put it in a txt file named api_key before implementing this code
# we are not using the api key directly here as it must not be exposed on a public platform
# so we are reading it from a .txt file which is ignored while commiting to the changes
# Id of channel = 'https://www.googleapis.com/youtube/v3/channels?key={'+api_key+'}&forUsername='+username+'&part=id'

# part1-get a list of video ids
# function for making a list of urls


def get_all_video_links_in_channel(channel_id):

    base_video_url = 'https://www.youtube.com/watch?v='
    base_search_url = 'https://www.googleapis.com/youtube/v3/search?'

    first_url = base_search_url+'key={}&channelId={}&part=snippet,id&order=date&maxResults=25'.format(api_key, channel_id)

    video_links = []
    url = first_url
    count = 0
    while(count < 4):                            # we want to download 4 images from each Channel
        inp = urllib.request.urlopen(url)
        resp = json.load(inp)

        for i in resp['items']:
            if i['id']['kind'] == "youtube#video":
                video_links.append(base_video_url + i['id']['videoId'])
                count += 1

        try:
            next_page_token = resp['nextPageToken']
            url = first_url + '&pageToken={}'.format(next_page_token)
        # The above 3 lines should be used when you want to download many thumbnails from a channels
        # the above code navigates to the next page of the same channel

        except:
            break


    return video_links
# mkbhdlinks =[]


# function for downloading thumbnails
def download(url, file_name):
        # open in binary mode
        with open(file_name, "wb") as file:
            # get request
            response = get(url)
            # write to file
            file.write(response.content)


allChannelIds = []


with open('/channel_list_top_500.csv') as f:
    reader = csv.reader(f, delimiter=',')
    aci = list(reader)

for j in range(len(aci)):
    allChannelIds.append(aci[j][0])    

print('The number of channels are as under')
print(len(allChannelIds))
# allChannelIds = pd.read_csv('./channel_list_top_500.csv',sep=',')

# Making a new folder to save images in
path = "./Good"          ##making it the new directory to store the good thumbnails
if not os.path.isdir(path):
	os.makedirs(path)

mylinks = []
for i in range(len(allChannelIds)):
    mylinks = get_all_video_links_in_channel(allChannelIds[i])
    # print(mylinks)
    playlist = []
    print("i = ", i)

    for r in range(len(mylinks)):
        segments = mylinks[r].rpartition('v=')
        playlist.append(segments[2])
        # print(segments[2])
    # print(playlist)

    # part 2- downloading thumbnails
    name = []

    for r in range(len(playlist)):
        name.append(r)

    name1 = [str(item) for item in name]    
    # print(name1)

    for r in range(len(playlist)):
        url = 'https://img.youtube.com/vi/'+playlist[r]+'/hqdefault.jpg'
        print(url)
        filename = './good'+str(i)+'_'+name1[r]+'.jpg'
        # urllib.request.urlretrieve(url, r)
        download(url, filename)
        print("r=", r)
    print(allChannelIds[i])
