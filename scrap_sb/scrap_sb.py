import urllib3
from BeautifulSoup import BeautifulSoup
import requests
import pandas as pd
import csv

# This code is to get the list of channels of a particular group
# Different list types are available in the comments.
# Accordingly convert them and run them to get different csv files
# Remember to change the csv file's name at the name so that the list is not over written

###### list_1 ######
# To download the top 25 channels

# source = requests.get('https://socialblade.com/youtube/').text
# soup = BeautifulSoup(source,'lxml')
# row=[]
# all_links = soup.find_all('div',{'class':'table-cell section-mg'})
# print(all_links)

###### list_2 ######
# To download the top 500 youTube channels most subscribed
# Remember to check the box most subscribed on the social blade site

http = urllib3.PoolManager()

url = 'https://socialblade.com/youtube/top/trending/top-500-channels-1-day/most-subscribed'
source = http.request('GET', url)
soup = BeautifulSoup(source.data)



# source = requests.get('https://socialblade.com/youtube/top/trending/top-500-channels-1-day/most-subscribed').text
# soup = BeautifulSoup(source, 'lxml')
row = []
all_links = soup.find_all('div', {'style':'float: left; width: 350px; line-height: 25px;'})

# The common code for all the lists
for link in all_links:
	row.append(link.find('a'))

row2 = []
for link in row:
	row2.append(link.get('href'))
# print(row2)

channel_id = []

for i in range(200,len(row2)):

    channel = requests.get('https://socialblade.com'+row2[i]).text
    soup_channel = BeautifulSoup(channel, 'lxml')

    channel_a = []

    all_links1 = soup_channel.find_all('div', {'style': 'float: right;'})
# print(all_links1)
    for link in all_links1:
        channel_a.append(link.find('a'))

    string = str(channel_a)

    segments = string.rpartition('c/')

    # new_segments = segments[2].rpartition('" rel=')

    channel_id.append(segments[2])
    print('i=', i)

print(channel_id)
while '' in channel_id:                        # There are few empty strings in the list
    channel_id.remove('')                       # This line of code will simply delete it

csvfile = 'channel_List_top_50'           # Remember to change the name of the csv file for different lists

with open(csvfile, 'w') as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in channel_id:
        writer.writerow([val])

print('The number of channel ids extracted are ', len(channel_id))
