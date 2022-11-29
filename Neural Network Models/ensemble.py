from PIL import Image
import os
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt

import pandas as pd
import re
import tensorflow as tf
import matplotlib
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, Reshape
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.optimizers import Adam, SGD
import cv2
from pasta.augment import inline
# %matplotlib inline

framObjTrain = {'img': [],
                'mask': []
                }

framObjValidation = {'img': [],
                     'mask': []
                     }
framObjTrain2 = {'img': [],
                'mask': []
                }

## defining data Loader function
def LoadData(frameObj=None, imgPath=None, maskPath=None, shape1=256, shape2=256):
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

        img = cv2.resize(img, (shape1, shape2))
        mask = cv2.resize(mask, (shape1, shape2))

        frameObj['img'].append(img)
        frameObj['mask'].append(mask[:,:,0])

    return frameObj

framObjTrain = LoadData( framObjTrain, imgPath = 'D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/train',
                        maskPath = 'D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/GT', shape1 = 256, shape2=256)

framObjValidation  = LoadData( framObjValidation, imgPath = 'D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/original-val',
                               maskPath = 'D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/mask-val' , shape1 = 256, shape2=256)


def Conv2dBlock(inputTensor, numFilters, kernelSize=3, doBatchNorm=True):
    # first Conv
    x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(inputTensor)

    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation('relu')(x)

    # Second Conv
    x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(x)
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation('relu')(x)

    return x


# Now defining Unet
def GiveMeUnet(inputImage, numFilters=16, droupouts=0.2, doBatchNorm=True):
    # defining encoder Path
    c1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = tf.keras.layers.Dropout(droupouts)(p1)

    c2 = Conv2dBlock(p1, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.Dropout(droupouts)(p2)

    c3 = Conv2dBlock(p2, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = tf.keras.layers.Dropout(droupouts)(p3)

    c4 = Conv2dBlock(p3, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = tf.keras.layers.Dropout(droupouts)(p4)

    c5 = Conv2dBlock(p4, numFilters * 16, kernelSize=3, doBatchNorm=doBatchNorm)

    # defining decoder path
    u6 = tf.keras.layers.Conv2DTranspose(numFilters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.Dropout(droupouts)(u6)
    c6 = Conv2dBlock(u6, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)

    u7 = tf.keras.layers.Conv2DTranspose(numFilters * 4, (3, 3), strides=(2, 2), padding='same')(c6)

    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(droupouts)(u7)
    c7 = Conv2dBlock(u7, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)

    u8 = tf.keras.layers.Conv2DTranspose(numFilters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(droupouts)(u8)
    c8 = Conv2dBlock(u8, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)

    u9 = tf.keras.layers.Conv2DTranspose(numFilters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    u9 = tf.keras.layers.Dropout(droupouts)(u9)
    c9 = Conv2dBlock(u9, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)

    # output = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid')(c9)
    output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[inputImage], outputs=[output])
    return model

from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity



# segnet

framObjTrain2  = LoadData( framObjTrain2, imgPath = 'D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/original-val',
                               maskPath = 'D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/mask-val' , shape1 = 256, shape2=192)

# x_train, x_test, y_train, y_test = train_test_split(np.array(framObjTrain2['img']), np.array(framObjTrain2['mask']), test_size = 0.25, random_state = 101)

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

inputs = tf.keras.layers.Input((256, 256, 3))
unet3 = GiveMeUnet(inputs, droupouts=0.15)
unet3.compile(optimizer='Adam', loss='binary_crossentropy')

segnet = Model(inputs=img_input, outputs=pred)
segnet.compile(optimizer= SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False), loss= ["binary_crossentropy"])

segnet.load_weights('D:/Lancaster-MSC/Satellite-Image/code/segnet_100_epoch.h5')

unet = load_model('D:/Lancaster-MSC/Satellite-Image/code/unet-150pic.h5')
unet2 = load_model('D:/Lancaster-MSC/Satellite-Image/code/unet-140pic.h5')
unet3.load_weights('D:/Lancaster-MSC/Satellite-Image/code/unet-130pic.h5')


# voting

img_pred = segnet.predict(np.array(framObjTrain2['img']))
unet_150 = (unet.predict(np.array(framObjValidation['img'])).reshape(len(img_pred),256,256))
unet_140 = (unet2.predict(np.array(framObjValidation['img'])).reshape(len(img_pred),256,256))
unet_130 = (unet3.predict(np.array(framObjValidation['img'])).reshape(len(img_pred),256,256))
seg_pred = np.resize(img_pred, (len(img_pred),256,256))

def enhance3(prediction1,prediction2,prediction3):
    input = []
    for i in range(len(img_pred)):
      a = (prediction1[i].flatten())
      b = (prediction2[i].flatten())
      c = (prediction3[i].flatten())
      sum = (a+b+c)/3
      # for j in range(len(a)):
      #     sum += a[j]
      # ave = sum / len(a)
      #print(ave)
      for m in range(len(a)):
        if sum[m] > 0.05:
          sum[m] = 1
        else:
          sum[m] = 0
      input.append(sum)
    return input

def enhance2(prediction1,prediction2):
    input = []
    for i in range(len(img_pred)):
      a = (prediction1[i].flatten())
      b = (prediction2[i].flatten())

      sum = (a+b)/2
      # for j in range(len(a)):
      #     sum += a[j]
      # ave = sum / len(a)
      #print(ave)
      for m in range(len(a)):
        if sum[m] > 0.05:
          sum[m] = 1
        else:
          sum[m] = 0
      input.append(sum)
    return input

# print(seg_pred.shape)
def enhance(prediction):
    input = []
    for i in range(len(img_pred)):
      a = (prediction[i].flatten())
      sum = 0
      for j in range(len(a)):
          sum += a[j]
      ave = sum / len(a)
      #print(ave)
      for m in range(len(a)):
        if a[m] > ave:
          a[m] = 1
        else:
          a[m] = 0
      input.append(a)
    return input


mask = np.array(framObjValidation['mask'])
# mask = np.resize(y_test, (50,256,256))
output = enhance(mask)

input_u_130 = enhance(unet_130)
input_u_140 = enhance(unet_140)
input_u_150 = enhance(unet_150)
input_seg = enhance(seg_pred)

#input_final5 = enhance2(input_seg,input_u_150) # 0.8309310913085938

input_final = [] #96.69
# f1-score
#  0.8566986517546242
# recall
#  0.859362483864501
# precision
#  0.8806608174696575
# accuracy
#  0.9668991088867187
for i in range(len(img_pred)):
    input_final.append([])

input_final2 = [] # voting weighted 96.75
# f1-score
#  0.8591840886109355
# recall
#  0.8580738648200927
# precision
#  0.8901800206377497
# accuracy
#  0.9675228881835938

# 0.1,0.3,0.4,0.2
# f1-score
#  0.8663233411951038
# recall
#  0.8773717086122664
# precision
#  0.8767667110207391
# accuracy
#  0.9680438232421875

for i in range(len(img_pred)):
    input_final2.append([])

# input_final3 = enhance3(unet_130,unet_150,unet_140) #95.11
# # f1-score
# #  0.8332444136848095
# # recall
# #  0.9494453435208947
# # precision
# #  0.7514829995752379
# # accuracy
# #  0.9511248779296875
#
#
# input_final4 = [] #96.78
# # f1-score
# #  0.8695428668280581
# # recall
# #  0.8961197588088315
# # precision
# #  0.8623200052270231
# # accuracy
# #  0.9678387451171875
# for i in range(len(img_pred)):
#     input_final4.append([])
#
# for i in range(len(img_pred)):
#     for j in range(len(input_seg[i])):
#         add = 0
#         if input_seg[i][j] == 1:
#             add += 1
#         if input_u_150[i][j] == 1:
#             add += 1
#         if input_u_140[i][j] == 1:
#             add += 1
#
#         if add > 1:
#             input_final[i].append(1)
#         else:
#             input_final[i].append(0)
#
for i in range(len(img_pred)):
    for j in range(len(input_seg[i])):
        add = 0
        if input_seg[i][j] == 1:
            add += 0.1
        if input_u_150[i][j] == 1:
            add += 0.3
        if input_u_140[i][j] == 1:
            add += 0.4
        if input_u_130[i][j] == 1:
            add += 0.2
        if add > 0.5:
            input_final2[i].append(1)
        else:
            input_final2[i].append(0)
#
# for i in range(len(img_pred)):
#     for j in range(len(input_seg[i])):
#         add = 0
#         if input_u_130[i][j] == 1:
#             add += 1
#         if input_u_150[i][j] == 1:
#             add += 1
#         if input_u_140[i][j] == 1:
#             add += 1
#
#         if add > 1:
#             input_final4[i].append(1)
#         else:
#             input_final4[i].append(0)



# print('--------voting average---------')
# # 0.9669
# f1_sco = 0
# recall_sco = 0
# precision_sco = 0
# acc_sco = 0
#
# for i in range(len(input_final)):
#   acc_sco += accuracy_score(output[i], input_final5[i])
#   precision_sco += precision_score(output[i], input_final5[i])
#   recall_sco += recall_score(output[i], input_final5[i])
#
#
# recall_sco = recall_sco/len(input_final)
# precision_sco = precision_sco/len(input_final)
# acc_sco = acc_sco/len(input_final)
# f1_sco = (2*precision_sco*recall_sco)/(precision_sco + recall_sco)
# print('f1-score\n',f1_sco)
# print('recall\n',recall_sco)
# print('precision\n',precision_sco)
# print('accuracy\n',acc_sco)

# print('--------voting---------')
# # 0.9669
# f1_sco = 0
# recall_sco = 0
# precision_sco = 0
# acc_sco = 0
#
# for i in range(len(input_final)):
#   acc_sco += accuracy_score(output[i], input_final[i])
#   precision_sco += precision_score(output[i], input_final[i])
#   recall_sco += recall_score(output[i], input_final[i])
#
#

# recall_sco = recall_sco/len(input_final)
# precision_sco = precision_sco/len(input_final)
# acc_sco = acc_sco/len(input_final)
# f1_sco = (2*precision_sco*recall_sco)/(precision_sco + recall_sco)
# print('f1-score\n',f1_sco)
# print('recall\n',recall_sco)
# print('precision\n',precision_sco)
# print('accuracy\n',acc_sco)

print('--------voting2---------')

f1_sco = 0
recall_sco = 0
precision_sco = 0
acc_sco = 0

for i in range(len(input_final2)):
  acc_sco += accuracy_score(output[i], input_final2[i])
  precision_sco += precision_score(output[i], input_final2[i])
  recall_sco += recall_score(output[i], input_final2[i])


recall_sco = recall_sco/len(input_final)
precision_sco = precision_sco/len(input_final)
acc_sco = acc_sco/len(input_final)
f1_sco = (2*precision_sco*recall_sco)/(precision_sco + recall_sco)
print('f1-score\n',f1_sco)
print('recall\n',recall_sco)
print('precision\n',precision_sco)
print('accuracy\n',acc_sco)

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(output, input_final2)

plt.plot(fpr, tpr, label='ROC')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
#
# print('--------voting3---------')
# f1_sco = 0
# recall_sco = 0
# precision_sco = 0
# acc_sco = 0
#
# for i in range(len(input_final3)):
#   acc_sco += accuracy_score(output[i], input_final3[i])
#   precision_sco += precision_score(output[i], input_final3[i])
#   recall_sco += recall_score(output[i], input_final3[i])
#
#
# recall_sco = recall_sco/len(input_seg)
# precision_sco = precision_sco/len(input_seg)
# acc_sco = acc_sco/len(input_seg)
# f1_sco = (2*precision_sco*recall_sco)/(precision_sco + recall_sco)
# print('f1-score\n',f1_sco)
# print('recall\n',recall_sco)
# print('precision\n',precision_sco)
# print('accuracy\n',acc_sco)
#
# print('--------unet---------')
# f1_sco = 0
# recall_sco = 0
# precision_sco = 0
# acc_sco = 0
#
# for i in range(len(input_final)):
#   acc_sco += accuracy_score(output[i], input_final4[i])
#   precision_sco += precision_score(output[i], input_final4[i])
#   recall_sco += recall_score(output[i], input_final4[i])
#
# recall_sco = recall_sco/len(input_u_150)
# precision_sco = precision_sco/len(input_u_150)
# acc_sco = acc_sco/len(input_u_150)
# f1_sco = (2*precision_sco*recall_sco)/(precision_sco + recall_sco)
# print('f1-score\n',f1_sco)
# print('recall\n',recall_sco)
# print('precision\n',precision_sco)
# print('accuracy\n',acc_sco)