import numpy as np
import sys
from contextlib import redirect_stdout
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.models import load_model
import cv2
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

img_width,img_height = 224,224
test_data_dir='./th_data3/test'
nb_test_samples=128
seed=7
batch_size = 8
np.random.seed(seed)
#import numpy as np
#from  keras.models import  load_model
#from keras.preprocessing import image
#import cv2
#from keras.applications.vgg16 import preprocess_input

#from keras.applications.resnet50 import ResNet50 
print('saving bottleneck_features...')
datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
model = applications.VGG16(include_top=False, weights = 'imagenet')
generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
print('2.1')
bottleneck_features_test = model.predict_generator(
    generator, nb_test_samples // batch_size)
print('2.2')
np.save(open('bottleneck_features_test_pred.npy', 'wb'),
        bottleneck_features_test)

#datagen = ImageDataGenerator(rescale=1./255)


test_data = np.load(open('bottleneck_features_test_pred.npy', 'rb'))

#model =  ResNet50(weights = 'imagenet')
model =load_model('second_model.h5')
#img_path = './th_data3/test/good/1.jpg'
#im = cv2.resize(cv2.imread(img_path), (150,150)).astype(np.float32)
#im[:,:,0] -= 103.939
#im[:,:,1] -= 116.779
#im[:,:,2] -= 123.68
#im = im.transpose((2,0,1))
#im = np.expand_dims(im, axis=0)
#img = image.load_img(img_path,target_size=(150,150))
#x = image.img_to_array(img)
#print(x.shape)
#x = np.expand_dims(x,axis=0)
#print(x.shape)
##x.setflags(write=1)
#x=preprocess_input(x)
##print(x.shape)
pred=model.predict(test_data)
pred_labels=[]
for i in range(len(pred)):
    if pred[i]>=0.5 :
        pred_labels.append(0)
    else:
        pred_labels.append(1)
print(pred_labels)
print('bad=',np.count_nonzero(pred_labels),'good=',len(pred)-np.count_nonzero(pred_labels))
#print(pred)
