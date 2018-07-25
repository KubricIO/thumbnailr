import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Concatenate
from keras import initializers, regularizers, applications
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
import datetime
import time
from PIL import Image
import cv2
import glob
import shutil
from keras import backend as K

# dimensions of our images.
img_width, img_height = 150, 150
start = time.clock()

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = './mark_1/train'
validation_data_dir = './mark_1/val'
test_data_dir = './mark_1/test'


def get_filecount(path_to_directory):
    if os.path.exists(path_to_directory):
        path, dirs, files = os.walk(path_to_directory).__next__()
        file_count = len(files)
        return file_count
    else:
        print("path does not exist")
        return 0


# def round_off(dirc, no_dp):
#     list_of_files = glob.glob(dirc + '/*')
#     latest_file = max(list_of_files, key=os.path.getctime)
#     print(latest_file)
#     for i in range(no_dp):
#         shutil.copy(latest_file, dirc + '/' + 'dub' + str(i + 1) + '.jpg')
#         i += 1
#     return


epochs = 1
batch_size = 8

nb_train_1_samples = get_filecount("mark_1/train/rate1")
nb_train_2_samples = get_filecount("mark_1/train/rate2")
nb_train_3_samples = get_filecount("mark_1/train/rate3")

nb_train_1_samples = nb_train_1_samples - nb_train_1_samples % batch_size
nb_train_2_samples = nb_train_2_samples - nb_train_2_samples % batch_size
nb_train_3_samples = nb_train_3_samples - nb_train_3_samples % batch_size
nb_train_samples = nb_train_1_samples + nb_train_2_samples + nb_train_3_samples


# nb_val_1_samples = get_filecount("mark_1/val/rate1")
# nb_val_2_samples = get_filecount("mark_1/val/rate2")
# nb_val_3_samples = get_filecount("mark_1/val/rate3")
#
# nb_val_1_samples = nb_val_1_samples - nb_val_1_samples % batch_size
# nb_val_2_samples = nb_val_2_samples - nb_val_2_samples % batch_size
# nb_val_3_samples = nb_val_3_samples - nb_val_3_samples % batch_size
# nb_validation_samples = nb_val_1_samples + nb_val_2_samples + nb_val_3_samples


nb_test_1 = get_filecount("mark_1/test/rate1")
nb_test_2 = get_filecount("mark_1/test/rate2")
nb_test_3 = get_filecount("mark_1/test/rate3")

nb_test_1 = nb_test_1 - nb_test_1 % batch_size
nb_test_2 = nb_test_2 - nb_test_2 % batch_size
nb_test_3 = nb_test_3 - nb_test_3 % batch_size
nb_test_samples = nb_test_1 + nb_test_2 + nb_test_3

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def myFunc(image):
    image = np.array(image)
    lab_image = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
    return Image.fromarray(lab_image)

def save_bottlebeck_features():
    print('\n\nsaving bottleneck_features...')
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 horizontal_flip=True)

    # build the VGG16 network
    model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=3)
    print('1, resnet_v2 model has been loaded\n')

    # For the training data
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print("Generating train bottleneck features")

    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)

    print("saving train bottleneck features")

    np.save(open('bottleneck_features_train_3class_resnet_kf.npy', 'wb'),
            bottleneck_features_train)
    print("bottleneck features for the training data has been stored")

    # # for the validation data
    # generator = datagen.flow_from_directory(
    #     validation_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     shuffle=False)
    #
    # print("Generating validation bottleneck features")
    #
    # bottleneck_features_validation = model.predict_generator(
    #     generator, nb_validation_samples // batch_size)
    #
    # print("Generating validation bottleneck features")
    #
    # np.save(open('bottleneck_features_validation_3class_resnet.npy', 'wb'),
    #         bottleneck_features_validation)
    # print("bottleneck features for the validation data has been stored")

    # For the test data
    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print("Generating test bottleneck features")

    bottleneck_features_test = model.predict_generator(
        generator, nb_test_samples // batch_size)

    print("Generating test bottleneck features")

    np.save(open('bottleneck_features_test_3class_resnet_kf.npy', 'wb'),
            bottleneck_features_test)
    print("bottleneck features for the test data has been stored")


def train_top_model():
    #print('training model...')
    train_data = np.load(open('bottleneck_features_train_3class_resnet_kf.npy', 'rb'))
    train_labels = np.array([0] * int(nb_train_1_samples) + [1] * int(nb_train_2_samples) + [2] * int(nb_train_3_samples))
    tr_labels = np.array([0] * int(nb_train_1_samples) + [1] * int(nb_train_2_samples) + [2] * int(nb_train_3_samples))
    #validation_data = np.load(open('bottleneck_features_validation_3class_resnet.npy', 'rb'))
    #validation_labels = np.array([0] * int(nb_val_1_samples) + [1] * int(nb_val_2_samples) + [2] * int(nb_val_3_samples))

    test_data = np.load(open('bottleneck_features_test_3class_resnet_kf.npy', 'rb'))
    test_labels = np.array([0] * int(nb_test_1) + [1] * int(nb_test_2) + [2] * int(nb_test_3))
    #train_labels = to_categorical(train_labels, 3)
    #validation_labels = to_categorical(validation_labels, 3)
    #test_labels = to_categorical(test_labels, 3)


    kfold = StratifiedKFold(n_splits=1, shuffle=True, random_state=seed)
    for train, test in kfold.split(train_data, tr_labels):
        print(train)
        print(test)
        model = Sequential()
        # Inception Model

        # Block1 = Sequential()
        # Block2 = Sequential()
        # Block3 = Sequential()
        # Block4 = Sequential()
        #
        # Block1.add(Conv2D(256, (1, 1), activation='relu', padding='same',input_shape=train_data.shape[1:]))
        # Block1.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        #
        # Block2.add(Conv2D(256, (1, 1), activation='relu', padding='same',input_shape=train_data.shape[1:]))
        # Block2.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
        #
        # Block3.add(Conv2D(128, (1, 1), activation='relu', padding='same', input_shape=train_data.shape[1:]))
        #
        # Block4.add(Conv2D(256, (1, 1), activation='relu', padding='same', input_shape=train_data.shape[1:]))
        # Block4.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'))
        #
        # model.add(Concatenate([Block1, Block2, Block3, Block4],input_shape=train_data.shape[1:]))

        # model.add(merged ,input_shape=train_data.shape[1:])

        # Inception over
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(4096, kernel_initializer=initializers.glorot_uniform(seed=None), kernel_regularizer=regularizers.l2(0.01),
                  activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(3, activation='softmax'))
        print('3')
        # checkpointer = ModelCheckpoint(filepath='model_class3_resnetv2_dense1_4096_2_kf.h5', verbose=1, save_best_only=True)
        # callbacks_list = [checkpointer]
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
        # optimizer = sgd
        optimizer = adam
        #model=load_model('model_class3_resnetv2_dense1_4096_2.h5')
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # print(len(train_data))
        # print(len(train_labels))
        # print(len(validation_data))
        # print(len(validation_labels))

        print("shape of the model output = ", model.output_shape)
        # train_labels = to_categorical(train_labels, 3)
        model.fit(train_data[train], train_labels[train],
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(train_data[test], train_labels[test]))
        # model.save_weights(top_model_weights_path)
            #name = 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        #model.save('model_test2.h5')
    scores = model.evaluate(test_data, test_labels,
                            batch_size=batch_size,
                            verbose=2,
                            sample_weight=None,
                            steps=None)
    print(scores)

    #
    # scores1 = model.predict(test_data, batch_size=batch_size, verbose=2)
    # print("\n\n")
    # print(scores1)
    # print("\n\n")
    # print(scores)
    # print("\n\n")
    # diff = scores - scores1
    # print(diff)
    # print("test_acc: ", "%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    #
    # print('4 : Done and Dusted')


#save_bottlebeck_features()
train_top_model()
print("\n\ntime taken =", time.clock() - start)
