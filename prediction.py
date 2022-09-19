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

## defining data Loader function
def LoadData(frameObj=None, imgPath=None, shape1=256, shape2=256):
    imgNames = os.listdir(imgPath)

    imgAddr = imgPath + '/'

    for i in range(len(imgNames)):
        img = plt.imread(imgAddr + imgNames[i])

        img = cv2.resize(img, (shape1, shape2))

        frameObj['img'].append(img)

    return frameObj

from keras.models import load_model

# segnet

def enhance(prediction,img_pred):
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

segnet = Model(inputs=img_input, outputs=pred)
segnet.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False), loss=["binary_crossentropy"])

segnet.load_weights('D:/Lancaster-MSC/Satellite-Image/code/segnet_100_epoch.h5')

unet = load_model('D:/Lancaster-MSC/Satellite-Image/code/unet-150pic.h5')
unet2 = load_model('D:/Lancaster-MSC/Satellite-Image/code/unet-140pic.h5')



def voting(data1,data2):

    # mask = np.array(framObjValidation['mask'])
    # # mask = np.resize(y_test, (50,256,256))
    # output = enhance(mask)
    img_pred = segnet.predict(data1)
    unet_150 = (unet.predict(data2).reshape(len(img_pred), 256, 256))
    unet_140 = (unet2.predict(data2).reshape(len(img_pred), 256, 256))
    seg_pred = np.resize(img_pred, (len(img_pred), 256, 256))

    # voting
    input_u_140 = enhance(unet_140,img_pred)
    input_u_150 = enhance(unet_150,img_pred)
    input_seg = enhance(seg_pred,img_pred)

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

    for i in range(len(img_pred)):
        for j in range(len(input_seg[i])):
            add = 0
            if input_seg[i][j] == 1:
                add += 1
            if input_u_150[i][j] == 1:
                add += 1
            if input_u_140[i][j] == 1:
                add += 1

            if add > 1:
                input_final[i].append(1)
            else:
                input_final[i].append(0)
    return np.array(input_final)

val = []
train = []

framObjValidation1 = {'img': []
                     }
framObjTrain1 = {'img': []
                }

val.append(LoadData(framObjValidation1,
                             imgPath='D:/Lancaster-MSC/Satellite-Image/lake-decrease/Cameroun',
                              shape1=256, shape2=256))

train.append(LoadData(framObjTrain1, imgPath='D:/Lancaster-MSC/Satellite-Image/lake-decrease/Cameroun',
                          shape1=256,shape2=192))
# --------------------------

framObjValidation2 = {'img': []
                     }
framObjTrain2 = {'img': []
                }
val.append(LoadData(framObjValidation2,
                             imgPath='D:/Lancaster-MSC/Satellite-Image/lake-decrease/Chad',
                              shape1=256, shape2=256))

train.append(LoadData(framObjTrain2, imgPath='D:/Lancaster-MSC/Satellite-Image/lake-decrease/Chad',
                          shape1=256,shape2=192))
# ------------------------

framObjValidation3 = {'img': []
                     }
framObjTrain3 = {'img': []
                }
val.append(LoadData(framObjValidation3,
                             imgPath='D:/Lancaster-MSC/Satellite-Image/lake-decrease/Nigeria',
                              shape1=256, shape2=256))

train.append(LoadData(framObjTrain3, imgPath='D:/Lancaster-MSC/Satellite-Image/lake-decrease/Nigeria',
                          shape1=256,shape2=192))
# --------------------------

framObjValidation4 = {'img': []
                     }
framObjTrain4 = {'img': []
                }
val.append(LoadData(framObjValidation4,
                             imgPath='D:/Lancaster-MSC/Satellite-Image/lake-increase/Brazil',
                              shape1=256, shape2=256))

train.append(LoadData(framObjTrain4, imgPath='D:/Lancaster-MSC/Satellite-Image/lake-increase/Brazil',
                          shape1=256,shape2=192))
# --------------------------

framObjValidation5 = {'img': []
                     }
framObjTrain5 = {'img': []
                }
val.append(LoadData(framObjValidation5,
                             imgPath='D:/Lancaster-MSC/Satellite-Image/lake-increase/Brazil-2',
                              shape1=256, shape2=256))

train.append(LoadData(framObjTrain5, imgPath='D:/Lancaster-MSC/Satellite-Image/lake-increase/Brazil-2',
                          shape1=256,shape2=192))
# --------------------------

# framObjValidation6 = {'img': []
#                      }
# framObjTrain6 = {'img': []
#                 }
# val.append(LoadData(framObjValidation6,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Armenia',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain6, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Armenia',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation7 = {'img': []
#                      }
# framObjTrain7 = {'img': []
#                 }
# val.append(LoadData(framObjValidation7,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Canada',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain7, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Canada',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation8 = {'img': []
#                      }
# framObjTrain8 = {'img': []
#                 }
# val.append(LoadData(framObjValidation8,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Canada-2',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain8, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Canada-2',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation9 = {'img': []
#                      }
# framObjTrain9 = {'img': []
#                 }
# val.append(LoadData(framObjValidation9,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Canada-3',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain9, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Canada-3',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation10 = {'img': []
#                      }
# framObjTrain10 = {'img': []
#                 }
# val.append(LoadData(framObjValidation10,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Canada-4',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain10, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Canada-4',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation11 = {'img': []
#                      }
# framObjTrain11 = {'img': []
#                 }
# val.append(LoadData(framObjValidation11,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Canada-5',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain11, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Canada-5',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation12 = {'img': []
#                      }
# framObjTrain12 = {'img': []
#                 }
# val.append(LoadData(framObjValidation12,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/China',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain12, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/China',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation13 = {'img': []
#                      }
# framObjTrain13 = {'img': []
#                 }
# val.append(LoadData(framObjValidation13,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/China-2',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain13, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/China-2',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation14 = {'img': []
#                      }
# framObjTrain14 = {'img': []
#                 }
# val.append(LoadData(framObjValidation14,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Egypt',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain14, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Egypt',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation15 = {'img': []
#                      }
# framObjTrain15 = {'img': []
#                 }
# val.append(LoadData(framObjValidation15,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Egypt-2',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain15, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Egypt-2',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation16 = {'img': []
#                      }
# framObjTrain16 = {'img': []
#                 }
# val.append(LoadData(framObjValidation16,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Estonia',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain16, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Estonia',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation17 = {'img': []
#                      }
# framObjTrain17 = {'img': []
#                 }
# val.append(LoadData(framObjValidation17,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Ethiopia',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain17, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Ethiopia',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation18 = {'img': []
#                      }
# framObjTrain18 = {'img': []
#                 }
# val.append(LoadData(framObjValidation18,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Ethiopia-2',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain18, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Ethiopia-2',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation19 = {'img': []
#                      }
# framObjTrain19 = {'img': []
#                 }
# val.append(LoadData(framObjValidation19,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Ethiopia-3',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain19, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Ethiopia-3',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation20 = {'img': []
#                      }
# framObjTrain20 = {'img': []
#                 }
# val.append(LoadData(framObjValidation20,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Ethiopia-4',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain20, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Ethiopia-4',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation21 = {'img': []
#                      }
# framObjTrain21 = {'img': []
#                 }
# val.append(LoadData(framObjValidation21,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Ethiopia-5',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain21, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Ethiopia-5',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation22 = {'img': []
#                      }
# framObjTrain22 = {'img': []
#                 }
# val.append(LoadData(framObjValidation22,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Germany',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain22, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Germany',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation23 = {'img': []
#                      }
# framObjTrain23 = {'img': []
#                 }
# val.append(LoadData(framObjValidation23,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Hungary',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain23, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Hungary',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation24 = {'img': []
#                      }
# framObjTrain24 = {'img': []
#                 }
# val.append(LoadData(framObjValidation24,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Israel',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain24, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Israel',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation25 = {'img': []
#                      }
# framObjTrain25 = {'img': []
#                 }
# val.append(LoadData(framObjValidation25,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Israel-2',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain25, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Israel-2',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation26 = {'img': []
#                      }
# framObjTrain26 = {'img': []
#                 }
# val.append(LoadData(framObjValidation26,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Japan',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain26, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Japan',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation27 = {'img': []
#                      }
# framObjTrain27 = {'img': []
#                 }
# val.append(LoadData(framObjValidation27,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Japan-2',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain27, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Japan-2',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation28 = {'img': []
#                      }
# framObjTrain28 = {'img': []
#                 }
# val.append(LoadData(framObjValidation28,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Kenya',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain28, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Kenya',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation29 = {'img': []
#                      }
# framObjTrain29 = {'img': []
#                 }
# val.append(LoadData(framObjValidation29,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Malawi',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain29, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Malawi',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation30 = {'img': []
#                      }
# framObjTrain30 = {'img': []
#                 }
# val.append(LoadData(framObjValidation30,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Mongolia',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain30, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Mongolia',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation31 = {'img': []
#                      }
# framObjTrain31 = {'img': []
#                 }
# val.append(LoadData(framObjValidation31,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/New Zealand',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain31, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/New Zealand',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation32 = {'img': []
#                      }
# framObjTrain32 = {'img': []
#                 }
# val.append(LoadData(framObjValidation32,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Russia',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain32, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Russia',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation33 = {'img': []
#                      }
# framObjTrain33 = {'img': []
#                 }
# val.append(LoadData(framObjValidation33,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Russia-2',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain33, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Russia-2',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation34 = {'img': []
#                      }
# framObjTrain34 = {'img': []
#                 }
# val.append(LoadData(framObjValidation34,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Rwanda',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain34, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Rwanda',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation35 = {'img': []
#                      }
# framObjTrain35 = {'img': []
#                 }
# val.append(LoadData(framObjValidation35,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Sweden',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain35, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Sweden',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation36 = {'img': []
#                      }
# framObjTrain36 = {'img': []
#                 }
# val.append(LoadData(framObjValidation36,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Switzerland',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain36, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Switzerland',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation37 = {'img': []
#                      }
# framObjTrain37 = {'img': []
#                 }
# val.append(LoadData(framObjValidation37,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Switzerland-2',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain37, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Switzerland-2',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation38 = {'img': []
#                      }
# framObjTrain38 = {'img': []
#                 }
# val.append(LoadData(framObjValidation38,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Syria',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain38, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Syria',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation39 = {'img': []
#                      }
# framObjTrain39 = {'img': []
#                 }
# val.append(LoadData(framObjValidation39,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Tanzania',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain39, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Tanzania',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation40 = {'img': []
#                      }
# framObjTrain40 = {'img': []
#                 }
# val.append(LoadData(framObjValidation40,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Tanzania-2',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain40, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Tanzania-2',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation41 = {'img': []
#                      }
# framObjTrain41 = {'img': []
#                 }
# val.append(LoadData(framObjValidation41,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Turkey',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain41, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Turkey',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation42 = {'img': []
#                      }
# framObjTrain42 = {'img': []
#                 }
# val.append(LoadData(framObjValidation42,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Turkey-2',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain42, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Turkey-2',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation43 = {'img': []
#                      }
# framObjTrain43 = {'img': []
#                 }
# val.append(LoadData(framObjValidation43,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Turkey-3',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain43, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Turkey-3',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation44 = {'img': []
#                      }
# framObjTrain44 = {'img': []
#                 }
# val.append(LoadData(framObjValidation44,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Uganda',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain44, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Uganda',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation45 = {'img': []
#                      }
# framObjTrain45 = {'img': []
#                 }
# val.append(LoadData(framObjValidation45,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Uganda-2',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain45, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Uganda-2',
#                           shape1=256,shape2=192))
# # --------------------------
#
# framObjValidation46 = {'img': []
#                      }
# framObjTrain46 = {'img': []
#                 }
# val.append(LoadData(framObjValidation46,
#                              imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Uganda-3',
#                               shape1=256, shape2=256))
#
# train.append(LoadData(framObjTrain46, imgPath='D:/Lancaster-MSC/Satellite-Image/unchanged/Uganda-3',
#                           shape1=256,shape2=192))
# # --------------------------

print(len(val))
print(len(val[0]['img']))
import csv
save_path = 'D:/Lancaster-MSC/Satellite-Image/predict_numpy/new/'
for i in range(len(val)):

    prediction = voting(np.array(train[i]['img']),np.array(val[i]['img']))

    save_out = save_path + str(i+1) + '.csv'

    f = open(save_out,'a',newline='')
    writer = csv.writer(f)
    for j in prediction:
        writer.writerow(j)
    f.close()


# plt.imshow(input_final[0].reshape(256,256))
# plt.show()
# plt.imshow(np.array(framObjValidation['mask'])[0])
# plt.show()