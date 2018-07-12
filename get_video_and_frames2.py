from __future__  import unicode_literals
import cv2
import os
import requests
import shutil
from skimage.measure import compare_ssim
from uuid import uuid4
# from utils import logger_util
# logger = logger_util.get_logger(__name__)
# import ffmpy
import youtube_dl
imageB = cv2.imread('black.jpg',0)

def download_file(url):
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    return local_filename
def create_directory(directory):
    """
        :param directory: folder path which needs to be created if not already present
        :return: nothing
    """
    if not os.path.exists(directory):
#        logger.info("Creating directory {}".format(directory))
        os.makedirs(directory)
def videoToFrames2(url , directory_uuid ,count):
#    logger.info("entering video to frames")
#     if 'www.youtube.com' in url :
#         ydl_opts = {}
#         with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#             ydl.download([url])
#             info_dict = ydl.extract_info(url, download=False)
#             video_url = info_dict.get("url", None)
#             video_id = info_dict.get("id", None)
#             video_title = info_dict.get('title', None)
#             # logger.info("video_title",video_title)
#             # logger.info("video_id",video_id)
#             # logger.info("video_url",video_url)
#             file1 = video_title + '-' +video_id + '.mp4'
#             # ff = ffmpy.FFmpeg(inputs={video_title + '-' +video_id + '.mkv': None},
#             #                 outputs={video_title + '-' +video_id + '.mp4': None})
#             # ff.run()
#             #
#             # file1 = video_title + '-' + video_id + '.mp4'
#     else:
#         file1 = download_file(url)
 #       logger.info('Entering videoToFrames....')
    file1 = url
    vidcap = cv2.VideoCapture(file1)
    success,image = vidcap.read()

    print (success)
    time = 0
    path = directory_uuid+"/images"          ##making the new directory
    # path = directory
    if not os.path.isdir(path):
        os.makedirs(path)
    cv2.imwrite(os.path.join(path + '/' + str(count + 1) + ".jpg"), image)
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,time)
        #(score, diff) = compare_ssim(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), imageB, full=True)

        #     cv2.imwrite(os.path.join(path + '/' + str(count+1) + ".jpg"), image)
        #     #save frame as JPEG file
        #     count += 1
        #     img_sc[int(score*1000)] =1
        success,image1 = vidcap.read()
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if success :
            (score, diff) = compare_ssim(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY),cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), full=True)
            if score < 0.90 :
                cv2.imwrite(os.path.join(path + '/' + str(count + 1) + ".jpg"), image1)
                image = image1
                count+=1
        #print ('Read a new frame: ', success)
            time += 200
  #  logger.info(count)
    return count
#videoToFrames()
# url = 'xyzueiu_yfvwey9hvjifw_jcvi.mp4'
# Video_to_frames1(url)
#logger.info('start downloading')


url_list = ['Facebook Slideshow _ Facebook for Business-7oEzeHB9YLw.mp4',
            'dhoni.mp4',
            'saty.mp4',
            'marvel.mp4',
            'sony.mp4',
            'scared.mp4',
            ]
total_count = 9191
for i in range(len(url_list)):
   total_count = videoToFrames2(url_list[i],'/home/karthik/thumbnailr',total_count)
   print(total_count)
#logger.info('finished downloading')