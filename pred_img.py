from __future__ import unicode_literals

import matplotlib
matplotlib.use('agg')
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.models import load_model
import cv2
import os
import requests
import shutil


def download_file(url):

	local_filename = url.split('/')[-1]
	r = requests.get(url, stream=True)
	with open(local_filename, 'wb') as f:
		shutil.copyfileobj(r.raw, f)

	return local_filename

def videoToFrames(url):
	print("entering video to frames")
	if 'www.youtube.com' in url :
		ydl_opts = {}
		with youtube_dl.YoutubeDL(ydl_opts) as ydl:
			ydl.download([url])
			info_dict = ydl.extract_info(url, download=False)
			video_id = info_dict.get("id", None)
			video_title = info_dict.get('title', None)
			file1 = video_title + '-' +video_id + '.mkv'
			# ff = ffmpy.FFmpeg(inputs={video_title + '-' +video_id + '.mkv': None},
			# 				  outputs={video_title + '-' +video_id + '.mp4': None})
			# ff.run()
            #
			# file1 = video_title + '-' + video_id + '.mp4'
	else:
		file1 = download_file(url)

	vidcap = cv2.VideoCapture(file1)
	success,image = vidcap.read()
	print (success)
	count = 0
	time = 0

	path = "./"+"frames"+"/images"          ##making the new directory
	# path = directory
	if not os.path.isdir(path):
		os.makedirs(path)

	success = True
	while success:
		vidcap.set(cv2.CAP_PROP_POS_MSEC,time)
		cv2.imwrite(os.path.join(path + '/' + str(count+1) + ".jpg"), image)
		#save frame as JPEG file
		success,image = vidcap.read()
		#print ('Read a new frame: ', success)
		count += 1
		time += 200
	print (count)
	return count

img_width,img_height = 150,150
seed=7
batch_size = 8
np.random.seed(seed)

url = 'https://storage.googleapis.com/videos-kubric/video-fd73b100-c3f1-47df-9950-bf4ea5af6fa4.mp4'
number_of_thumbnails = 5
directory_uuid = './' + 'frames'
if not os.path.isdir(directory_uuid):
    os.makedirs(directory_uuid)
nb_test_samples = videoToFrames(url)

model_path = "./"


def predImg(nb_test_samples, directory_uuid, number_of_thumbnails):

    test_data_dir = directory_uuid
    img_width, img_height = 150, 150  # For resizing the input images

    seed = 7
    batch_size = 8
    np.random.seed(seed)
    datagen = ImageDataGenerator(rescale=1. / 255)

    # building the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    logger.info('saving bottleneck_features...')
    logger.info('Checkpoint pred Img, 2.1')
    bottleneck_features_test = model.predict_generator(
        generator, nb_test_samples // batch_size)
    logger.info('Checkpoint pred Img, 2.2')
    np.save(open('bottleneck_features_test_pred.npy', 'wb'),
            bottleneck_features_test)

    # Saving the 7 x 7 x 512 matrix for the input frames using the pretrained weights of the VGG16 network
    test_data = np.load(open('bottleneck_features_test_pred.npy', 'rb'))
    # Loading the .h5 weights file which has been trained
    model = load_model(model_path)

    pred = model.predict(test_data)
    pred = np.array(pred)
    pred = pred.flatten()

    pred_labels = []
    # Taking 0.5 as the threshold for the images to be classified as good or bad and this is purely subjective
    for i in range(len(pred)):
        if pred[i] >= 0.5:  ## the threshold to be chosen is subjective
            pred_labels.append(0)
        else:
            pred_labels.append(1)

    # logger.info(pred_labels)
    logger.info('image with max softmax score ='.format(np.argmax(pred) + 1))

    sorted_pred = np.sort(pred)
    logger.info("Printing the sorted prediction scores for frames of the given videoid")
    logger.info(sorted_pred)

    top_index = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:number_of_thumbnails]
    top_index = [val + 1 for val in top_index]

    return (top_index)

predImg(nb_test_samples, directory_uuid, number_of_thumbnails)