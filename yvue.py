import urllib.request
import json
import numpy as np 
import urllib.request
import shutil
from requests import get
import pandas as pd
import csv
# from config import api_key
import os

api_key = 'AIzaSyBQflK2Y8-UCPlAzlbcfA9TUXozs9fC2iU'



# username = 'marquesbrownlee'

# idofchannel = 'https://www.googleapis.com/youtube/v3/channels?key={'+api_key+'}&forUsername='+username+'&part=id'


# part1-get a list of video ids
# function for making a list of urls
def get_all_video_in_channel(channel_id):
    

    base_video_url = 'https://www.youtube.com/watch?v='
    base_search_url = 'https://www.googleapis.com/youtube/v3/search?'

    first_url = base_search_url+'key={}&channelId={}&part=snippet,id&order=date&maxResults=25'.format(api_key, channel_id)

    video_links = []
    url = first_url
    count=0
    # while (count<6):
    count+=1
    inp = urllib.request.urlopen(url)
    resp = json.load(inp)
    # print("count=",count)
    for i in resp['items']:
        # print("i=",i)
        if i['id']['kind'] == "youtube#video":
            if len(video_links)<5 :
                video_links.append(base_video_url + i['id']['videoId'])
            else:
                break    

    # try:
    #     next_page_token = resp['nextPageToken']
    #     url = first_url + '&pageToken={}'.format(next_page_token)
    # except:
    #     break
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

# mkbhdId = 'UCBJycsmduvYEL83R_U4JriQ'
# myChannelId='UCruH1rzS6lv_wUEajKdVKrg'
allChannelIds=[]

# path = "./channelids"          ##making the new directory
# if not os.path.isdir(path):
#     os.makedirs(path)


with open('../new_top_500.csv') as f:
    reader = csv.reader(f,delimiter=',')
    aci = list(reader)






for j in range(len(aci)):
    allChannelIds.append(aci[j][0])    

print(allChannelIds)    

# allChannelIds = pd.read_csv('/home/rahul/channelids.csv',sep=',')
# print(allChannelIds)

for i in range(len(allChannelIds)):

    print("getting links..")
    mylinks=get_all_video_in_channel(allChannelIds[i]) 
    # print(mylinks)

    playlist=[]

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
        # print(url)
        filename = '/home/rahul/th_data/'+str(i)+'_'+name1[r]+'.jpg'
        # urllib.request.urlretrieve(url, r)
        download(url,filename)
        print("r=",r)
    # print(allChannelIds[i])


