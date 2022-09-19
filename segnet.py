import matplotlib
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, Reshape
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.optimizers import Adam, SGD
import tensorflow as tf
import pandas as pd
import glob
import PIL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pasta.augment import inline
# %matplotlib inline
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from warnings import filterwarnings

filterwarnings('ignore')
plt.rcParams["axes.grid"] = False
np.random.seed(101)

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# filelist_trainx = sorted(glob.glob('D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/train-total/*.jpg'), key=numericalSort)
# X_train = np.array([np.array(Image.open(fname)) for fname in filelist_trainx])
#
# filelist_trainy = sorted(glob.glob('D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/mask-total/*.png'), key=numericalSort)
# Y_train = np.array([np.array(Image.open(fname)) for fname in filelist_trainy])
#
# x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.25, random_state = 101)

framObjTrain = {'img': [],
                'mask': []
                }

framObjValidation = {'img': [],
                     'mask': []
                     }


## defining data Loader function
def LoadData(frameObj=None, imgPath=None, maskPath=None, shape1=192, shape2=256):
    imgNames = os.listdir(imgPath)
    maskNames = []

    ## generating mask names
    for mem in imgNames:
        maskNames.append(re.sub('\.jpg', '.png', mem))

    imgAddr = imgPath + '/'
    maskAddr = maskPath + '/'

    for i in range(len(imgNames)):
        img = plt.imread(imgAddr + imgNames[i])
        mask = plt.imread(maskAddr + maskNames[i])
        #print(mask.shape)
        img = cv2.resize(img, (shape1, shape2))
        mask = cv2.resize(mask, (shape1, shape2))

        frameObj['img'].append(img)
        frameObj['mask'].append(mask[:,:,0])

    return frameObj

framObjTrain = LoadData( framObjTrain, imgPath = 'D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/train-total',
                        maskPath = 'D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/mask-total', shape1 = 256, shape2 = 192)
x_train, x_test, y_train, y_test = train_test_split(np.array(framObjTrain['img']), np.array(framObjTrain['mask']), test_size = 0.25, random_state = 101)

def iou(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis = -1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def dice_coef(y_true, y_pred, smooth = 100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return K.mean(K.equal(y_true, K.round(y_pred)))

def random_rotation(x_image, y_image):
    rows_x,cols_x, chl_x = x_image.shape
    rows_y,cols_y = y_image.shape
    rand_num = np.random.randint(-40,40)
    M1 = cv2.getRotationMatrix2D((cols_x/2,rows_x/2),rand_num,1)
    M2 = cv2.getRotationMatrix2D((cols_y/2,rows_y/2),rand_num,1)
    x_image = cv2.warpAffine(x_image,M1,(cols_x,rows_x))
    y_image = cv2.warpAffine(y_image.astype('float32'),M2,(cols_y,rows_y))
    return x_image, y_image.astype('int')

def horizontal_flip(x_image, y_image):
    x_image = cv2.flip(x_image, 1)
    y_image = cv2.flip(y_image.astype('float32'), 1)
    return x_image, y_image.astype('int')


def segnet(epochs_num, savename):
    # Encoding layer
    img_input = Input(shape=(192, 256, 3))
    x = Conv2D(64, (3, 3), padding='same', name='conv1', strides=(1, 1))(img_input)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, (3, 3), padding='same', name='conv5')(x)
    x = BatchNormalization(name='bn5')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv6')(x)
    x = BatchNormalization(name='bn6')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv7')(x)
    x = BatchNormalization(name='bn7')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(512, (3, 3), padding='same', name='conv8')(x)
    x = BatchNormalization(name='bn8')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv9')(x)
    x = BatchNormalization(name='bn9')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv10')(x)
    x = BatchNormalization(name='bn10')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(512, (3, 3), padding='same', name='conv11')(x)
    x = BatchNormalization(name='bn11')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv12')(x)
    x = BatchNormalization(name='bn12')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv13')(x)
    x = BatchNormalization(name='bn13')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    # Decoding Layer
    x = UpSampling2D()(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv1')(x)
    x = BatchNormalization(name='bn14')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv2')(x)
    x = BatchNormalization(name='bn15')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv3')(x)
    x = BatchNormalization(name='bn16')(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv4')(x)
    x = BatchNormalization(name='bn17')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv5')(x)
    x = BatchNormalization(name='bn18')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv6')(x)
    x = BatchNormalization(name='bn19')(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv7')(x)
    x = BatchNormalization(name='bn20')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv8')(x)
    x = BatchNormalization(name='bn21')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv9')(x)
    x = BatchNormalization(name='bn22')(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv10')(x)
    x = BatchNormalization(name='bn23')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv11')(x)
    x = BatchNormalization(name='bn24')(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv12')(x)
    x = BatchNormalization(name='bn25')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(1, (3, 3), padding='same', name='deconv13')(x)
    x = BatchNormalization(name='bn26')(x)
    x = Activation('sigmoid')(x)
    pred = Reshape((192, 256))(x)

    model = Model(inputs=img_input, outputs=pred)

    model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False), loss=["binary_crossentropy"]
                  , metrics=[iou, dice_coef, precision, recall, accuracy])
    # model.summary()
    hist = model.fit(x_train, y_train, epochs=epochs_num, batch_size=15, validation_data=(x_test, y_test), verbose=1)

    model.save(savename)
    return model, hist

# model, hist = segnet(epochs_num= 100, savename= 'D:/Lancaster-MSC/Satellite-Image/code/segnet_100_epoch.h5')

# Encoding layer
img_input = Input(shape= (192, 256, 3))
x = Conv2D(64, (3, 3), padding='same', name='conv1',strides= (1,1))(img_input)
x = BatchNormalization(name='bn1')(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)
x = BatchNormalization(name='bn2')(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
x = BatchNormalization(name='bn3')(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
x = BatchNormalization(name='bn4')(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(256, (3, 3), padding='same', name='conv5')(x)
x = BatchNormalization(name='bn5')(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), padding='same', name='conv6')(x)
x = BatchNormalization(name='bn6')(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), padding='same', name='conv7')(x)
x = BatchNormalization(name='bn7')(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(512, (3, 3), padding='same', name='conv8')(x)
x = BatchNormalization(name='bn8')(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', name='conv9')(x)
x = BatchNormalization(name='bn9')(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', name='conv10')(x)
x = BatchNormalization(name='bn10')(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(512, (3, 3), padding='same', name='conv11')(x)
x = BatchNormalization(name='bn11')(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', name='conv12')(x)
x = BatchNormalization(name='bn12')(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', name='conv13')(x)
x = BatchNormalization(name='bn13')(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Dense(1024, activation = 'relu', name='fc1')(x)
x = Dense(1024, activation = 'relu', name='fc2')(x)
# Decoding Layer
x = UpSampling2D()(x)
x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv1')(x)
x = BatchNormalization(name='bn14')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv2')(x)
x = BatchNormalization(name='bn15')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv3')(x)
x = BatchNormalization(name='bn16')(x)
x = Activation('relu')(x)

x = UpSampling2D()(x)
x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv4')(x)
x = BatchNormalization(name='bn17')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv5')(x)
x = BatchNormalization(name='bn18')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv6')(x)
x = BatchNormalization(name='bn19')(x)
x = Activation('relu')(x)

x = UpSampling2D()(x)
x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv7')(x)
x = BatchNormalization(name='bn20')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv8')(x)
x = BatchNormalization(name='bn21')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv9')(x)
x = BatchNormalization(name='bn22')(x)
x = Activation('relu')(x)

x = UpSampling2D()(x)
x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv10')(x)
x = BatchNormalization(name='bn23')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv11')(x)
x = BatchNormalization(name='bn24')(x)
x = Activation('relu')(x)

x = UpSampling2D()(x)
x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv12')(x)
x = BatchNormalization(name='bn25')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(1, (3, 3), padding='same', name='deconv13')(x)
x = BatchNormalization(name='bn26')(x)
x = Activation('sigmoid')(x)
pred = Reshape((192,256))(x)

model_1 = Model(inputs=img_input, outputs=pred)
model_1.compile(optimizer= SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False), loss= ["binary_crossentropy"]
              , metrics=[iou, dice_coef, precision, recall, accuracy])

model_1.load_weights('D:/Lancaster-MSC/Satellite-Image/code/segnet_100_epoch.h5')

# IOU:       |   92.31  |
# Dice Coef: |   54.48  |
# Precision: |   83.26  |
# Recall:    |   78.15  |
# F1-score:  |   80.62  |
# Accuracy:  |   94.74  |
# Loss:      |   19.77  |

# print('\n~~~~~~~~~~~~~~~Stats after 100 epoch~~~~~~~~~~~~~~~~~~~')
# print('\n-------------On Train Set--------------------------\n')
# res = model_1.evaluate(x_train, y_train, batch_size= 18)
# print('________________________')
# print('IOU:       |   {:.2f}  |'.format(res[1]*100))
# print('Dice Coef: |   {:.2f}  |'.format(res[2]*100))
# print('Precision: |   {:.2f}  |'.format(res[3]*100))
# print('Recall:    |   {:.2f}  |'.format(res[4]*100))
# print('F1-score:  |   {:.2f}  |',(2*(res[4]*100)*(res[3]*100))/((res[4]*100)+(res[3]*100)))
# print('Accuracy:  |   {:.2f}  |'.format(res[5]*100))
# print("Loss:      |   {:.2f}  |".format(res[0]*100))
# print('________________________')
print('\n-------------On Test  Set--------------------------\n')
res = model_1.evaluate(x_test, y_test, batch_size= 18)
print('________________________')
print('IOU:       |   {:.2f}  |'.format(res[1]*100))
print('Dice Coef: |   {:.2f}  |'.format(res[2]*100))
print('Precision: |   {:.2f}  |'.format(res[3]*100))
print('Recall:    |   {:.2f}  |'.format(res[4]*100))
print('Accuracy:  |   {:.2f}  |'.format(res[5]*100))
print("Loss:      |   {:.2f}  |".format(res[0]*100))
print('________________________')
# print('\n-------------On validation Set---------------------\n')
# res = model_1.evaluate(x_test, y_test, batch_size= 18)
# print('________________________')
# print('IOU:       |   {:.2f}  |'.format(res[1]*100))
# print('Dice Coef: |   {:.2f}  |'.format(res[2]*100))
# print('Precision: |   {:.2f}  |'.format(res[3]*100))
# print('Recall:    |   {:.2f}  |'.format(res[4]*100))
# print('Accuracy:  |   {:.2f}  |'.format(res[5]*100))
# print("Loss:      |   {:.2f}  |".format(res[0]*100))
# print('________________________')


# def enhance(img):
#     sub = (model_1.predict(img.reshape(1,192,256,3))).flatten()
#
#     for i in range(len(sub)):
#         if sub[i] > 0.5:
#             sub[i] = 1
#         else:
#             sub[i] = 0
#     return sub
#
# img_num = 49
# img_pred = model_1.predict(x_test[img_num].reshape(1,192,256,3))
# plt.figure(figsize=(16,16))
# plt.subplot(1,3,1)
# plt.imshow(x_test[img_num])
# plt.title('Original Image')
# plt.subplot(1,3,2)
# plt.imshow(y_test[img_num], plt.cm.binary_r)
# plt.title('Ground Truth')
# plt.subplot(1,3,3)
# plt.imshow(img_pred.reshape(192, 256), plt.cm.binary_r)
# plt.title('Predicted Output')
# plt.show()
#
#
# plt.figure(figsize=(12,12))
# plt.suptitle('Comparing the Prediction after enhancement')
# plt.subplot(3,2,1)
# plt.imshow(y_test[21],plt.cm.binary_r)
# plt.title('Ground Truth')
# plt.subplot(3,2,2)
# plt.imshow(enhance(x_test[21]).reshape(192,256), plt.cm.binary_r)
# plt.title('Predicted')
# plt.subplot(3,2,3)
# plt.imshow(y_test[47],plt.cm.binary_r)
# plt.title('Ground Truth')
# plt.subplot(3,2,4)
# plt.imshow(enhance(x_test[47]).reshape(192,256), plt.cm.binary_r)
# plt.title('Predicted')
# plt.subplot(3,2,5)
# plt.imshow(y_test[36],plt.cm.binary_r)
# plt.title('Ground Truth')
# plt.subplot(3,2,6)
# plt.imshow(enhance(x_test[36]).reshape(192,256), plt.cm.binary_r)
# plt.title('Predicted')
# plt.show()



# from skimage.transform import resize as imresize
#
# img_pred = model_1.predict(x_test)
# # seg_pred = img_pred
# # print(seg_pred.shape)
# # for i in range(len(img_pred)):
# #     seg_pred[i] = np.resize(seg_pred[i], (256,256))
# #seg_pred = cv2.resize(img_pred, (50,256))
#
# seg_pred = np.resize(img_pred, (50,256,256))
# # seg_pred = np.array(seg_pred)
# print(seg_pred.shape)
#
# input = []
# for i in range(len(seg_pred)):
#   a = (seg_pred[i].flatten())
#   sum = 0
#   for j in range(len(a)):
#       sum += a[j]
#   ave = sum / len(a)
#   #print(ave)
#   for m in range(len(a)):
#     if a[m] > ave:
#       a[m] = 1
#     else:
#       a[m] = 0
#   input.append(a)
#
# output = []
# # mask = y_test
# mask = np.resize(y_test, (50,256,256))
# # for i in range(len(y_test)):
# #     mask[i] = np.resize(y_test[i], (256,256))
#
# for i in range(len(mask)):
#   a = (mask[i].flatten())
#   sum = 0
#   for j in range(len(a)):
#       sum += a[j]
#   ave = sum / len(a)
#   #print(ave)
#   for m in range(len(a)):
#     if a[m] > ave:
#       a[m] = 1
#     else:
#       a[m] = 0
#   output.append(a)
#
# plt.imshow(y_test[0])
# plt.show()
# plt.imshow(input[0].reshape(256,256))
# plt.show()
# plt.imshow(output[0].reshape(256,256))
# plt.show()
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.metrics import f1_score, precision_score, recall_score
# f1_sco = 0
# recall_sco = 0
# precision_sco = 0
# acc_sco = 0
#
# for i in range(len(input)):
#   acc_sco += accuracy_score(output[i], input[i])
#   precision_sco += precision_score(output[i], input[i])
#   recall_sco += recall_score(output[i], input[i])
#   f1_sco += f1_score(output[i], input[i])
#
# f1_sco = f1_sco/len(input)
# recall_sco = recall_sco/len(input)
# precision_sco = precision_sco/len(input)
# acc_sco = acc_sco/len(input)
# print('f1-score\n',f1_sco)
# #print('Classification report\n',accuracy(output, input)) #0.9672
# print('recall\n',recall_sco)
# print('precision\n',precision_sco)
# print('accuracy\n',acc_sco)

# source : https://www.kaggle.com/code/hashbanger/skin-lesion-segmentation-using-segnet/notebook