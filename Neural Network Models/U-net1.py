import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

framObjTrain = {'img': [],
                'mask': []
                }

framObjValidation = {'img': [],
                     'mask': []
                     }


## defining data Loader function
def LoadData(frameObj=None, imgPath=None, maskPath=None, shape=256):
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

        img = cv2.resize(img, (shape, shape))
        mask = cv2.resize(mask, (shape, shape))

        frameObj['img'].append(img)
        frameObj['mask'].append(mask[:,:,0])

    return frameObj

framObjTrain = LoadData( framObjTrain, imgPath = 'D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/train',
                        maskPath = 'D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/GT', shape = 256)

framObjValidation  = LoadData( framObjValidation, imgPath = 'D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/original-val',
                               maskPath = 'D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/mask-val' , shape = 256)

# ## displaying data loaded by our function
# plt.subplot(1,2,1)
# plt.imshow(framObjTrain['img'][1])
# plt.subplot(1,2,2)
# plt.imshow(framObjTrain['mask'][1])
# plt.show()
# print(framObjTrain['img'][1].shape)
# print(framObjTrain['mask'][1].shape)
#
# this block essentially performs 2 convolution

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

from keras import backend as K
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

# ## instanctiating model
inputs = tf.keras.layers.Input((256, 256, 3))
unet = GiveMeUnet(inputs, droupouts=0.15)
unet.compile(optimizer='Adam', loss='binary_crossentropy', metrics=[iou, dice_coef, precision, recall, accuracy])
# train = np.array(framObjTrain['img'])[0:130]
# test = np.array(framObjTrain['mask'])[0:130]
# retVal = unet.fit(train, test,
#                   epochs = 150, shuffle = True,
#                   verbose = 0)


# save_path = 'D:/Lancaster-MSC/Satellite-Image/code/unet-130pic.h5'
# unet.save(save_path)

# plt.plot(retVal.history['loss'], label = 'training_loss')
# plt.plot(retVal.history['accuracy'], label = 'training_accuracy')
# plt.legend()
# plt.grid(True)



unet.load_weights('D:/Lancaster-MSC/Satellite-Image/code/unet-130pic.h5')


res = unet.evaluate(np.array(framObjValidation['img']), np.array(framObjValidation['mask']), batch_size= 18)
print('________________________')
print('IOU:       |   {:.2f}  |'.format(res[1]*100))
print('Dice Coef: |   {:.2f}  |'.format(res[2]*100))
print('Precision: |   {:.2f}  |'.format(res[3]*100))
print('Recall:    |   {:.2f}  |'.format(res[4]*100))
print('F1-score:  |   {:.2f}  |',(2*(res[4]*100)*(res[3]*100))/((res[4]*100)+(res[3]*100)))
print('Accuracy:  |   {:.2f}  |'.format(res[5]*100))
print("Loss:      |   {:.2f}  |".format(res[0]*100))
print('________________________')

# 150
# IOU:       |   95.47  |
# Dice Coef: |   85.72  |
# Precision: |   92.33  |
# Recall:    |   83.59  |
# F1-score:  |   87.74  |
# Accuracy:  |   95.83  |
# Loss:      |   12.29  |

#140
# IOU:       |   95.81  |
# Dice Coef: |   85.66  |
# Precision: |   92.92  |
# Recall:    |   84.18  |
# F1-score:  | | 88.33428953035443
# Accuracy:  |   96.09  |
# Loss:      |   10.68  |

# 130
# IOU:       |   94.75  |
# Dice Coef: |   82.40  |
# Precision: |   87.36  |
# Recall:    |   83.17  |
# F1-score:  85.2174571708
# Accuracy:  |   95.12  |
# Loss:      |   12.99  |

# ## function for getting 16 predictions
# def predict16(valMap, model, shape=256):
#     ## getting and proccessing val data
#     img = valMap['img']
#     mask = valMap['mask']
#     mask = mask[0:16]
#
#     imgProc = img[0:16]
#     imgProc = np.array(img)
#
#
#     predictions = model.predict(imgProc)
#     # for i in range(len(predictions)):
#     #     predictions[i] = cv2.merge((predictions[i, :, :, 0], predictions[i, :, :, 1], predictions[i, :, :, 2]))
#
#     return predictions, imgProc, mask
#
#
# def Plotter(img, predMask, groundTruth):
#     plt.figure(figsize=(7, 7))
#
#     plt.subplot(1, 3, 1)
#     plt.imshow(img)
#     plt.title('image')
#
#     plt.subplot(1, 3, 2)
#     plt.imshow(predMask)
#     plt.title('Predicted Mask')
#
#     plt.subplot(1, 3, 3)
#     plt.imshow(groundTruth)
#     plt.title('actual Mask')
#
#     plt.show()
#
#
# # def Plotter(img, predMask, groundTruth):
# #     plt.figure(figsize=(9, 9))
# #
# #     plt.subplot(1, 4, 1)
# #     plt.imshow(img)
# #     plt.title('image')
# #
# #     plt.subplot(1, 4, 2)
# #     plt.imshow(predMask)
# #     plt.title('Predicted Mask')
# #
# #     plt.subplot(1, 4, 3)
# #     plt.imshow(groundTruth)
# #     plt.title('actual Mask')
# #
# #     imh = predMask
# #     imh[imh < 0.5] = 0
# #     imh[imh > 0.5] = 1
# #
# #     plt.subplot(1, 4, 4)
# #     plt.imshow(cv2.merge((imh, imh, imh)) * img)
# #     plt.title('segmented Image')
# #
# #     plt.show()
#
# sixteenPrediction, actuals, masks = predict16(framObjValidation, unet)
#
# Plotter(actuals[1], sixteenPrediction[1][:,:,0], masks[1])
# Plotter(actuals[3], sixteenPrediction[3][:,:,0], masks[3])
# Plotter(actuals[4], sixteenPrediction[4][:,:,0], masks[4])
# Plotter(actuals[6], sixteenPrediction[6][:,:,0], masks[6])
# Plotter(actuals[7], sixteenPrediction[7][:,:,0], masks[7])
# Plotter(actuals[9], sixteenPrediction[9][:,:,0], masks[9])
# Plotter(actuals[5], sixteenPrediction[5][:,:,0], masks[5])
# Plotter(actuals[8], sixteenPrediction[8][:,:,0], masks[8])
# Plotter(actuals[11], sixteenPrediction[11][:,:,0], masks[11])
# Plotter(actuals[14], sixteenPrediction[14][:,:,0], masks[14])