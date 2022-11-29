from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
# from osgeo import gdal
from PIL import Image
import matplotlib.pyplot as plot
import random
import seaborn as sns
from keras.models import load_model
from keras.models import Model
from tensorflow.keras.optimizers import Adam
#from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
#from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, Activation, Add, Conv2DTranspose
from keras.applications.vgg16 import VGG16




def LoadImage(name, path):
    img = Image.open(os.path.join(path, name))
    img = np.array(img)

    image = img[:, :256]
    mask = img[:, 256:]

    return image, mask


def bin_image(mask):
    bins = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240])
    new_mask = np.digitize(mask, bins)
    return new_mask

width = 256
height = 256
classes = 2
batch_size = 10
def getSegmentationArr(image, classes, width=width, height=height):
    seg_labels = np.zeros((height, width, classes))
    img = image[:, :, 0]

    for c in range(classes):
        seg_labels[:, :, c] = (img == c).astype(int)
    return seg_labels


def give_color_to_seg_img(seg, n_classes=2):
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:, :, 0] += (segc * (colors[c][0]))
        seg_img[:, :, 1] += (segc * (colors[c][1]))
        seg_img[:, :, 2] += (segc * (colors[c][2]))

    return (seg_img)


def DataGenerator(path, batch_size=10, classes=13):
    files = os.listdir(path)
    while True:
        for i in range(0, len(files), batch_size):
            batch_files = files[i: i + batch_size]
            imgs = []
            segs = []
            for file in batch_files:
                # file = random.sample(files,1)[0]
                image, mask = LoadImage(file, path)
                mask_binned = bin_image(mask)
                labels = getSegmentationArr(mask_binned, classes)

                imgs.append(image)
                segs.append(labels)

            yield np.array(imgs), np.array(segs)



def fcn(vgg, classes=13, fcn8=False, fcn16=False):
    pool5 = vgg.get_layer('block5_pool').output
    pool4 = vgg.get_layer('block4_pool').output
    pool3 = vgg.get_layer('block3_pool').output

    conv_6 = Conv2D(1024, (7, 7), activation='relu', padding='same', name="conv_6")(pool5)
    conv_7 = Conv2D(1024, (1, 1), activation='relu', padding='same', name="conv_7")(conv_6)

    conv_8 = Conv2D(classes, (1, 1), activation='relu', padding='same', name="conv_8")(pool4)
    conv_9 = Conv2D(classes, (1, 1), activation='relu', padding='same', name="conv_9")(pool3)

    deconv_7 = Conv2DTranspose(classes, kernel_size=(2, 2), strides=(2, 2))(conv_7)
    add_1 = Add()([deconv_7, conv_8])
    deconv_8 = Conv2DTranspose(classes, kernel_size=(2, 2), strides=(2, 2))(add_1)
    add_2 = Add()([deconv_8, conv_9])
    deconv_9 = Conv2DTranspose(classes, kernel_size=(8, 8), strides=(8, 8))(add_2)

    if fcn8:
        output_layer = Activation('softmax')(deconv_9)
    elif fcn16:
        deconv_10 = Conv2DTranspose(classes, kernel_size=(16, 16), strides=(16, 16))(add_1)
        output_layer = Activation('softmax')(deconv_10)
    else:
        deconv_11 = Conv2DTranspose(classes, kernel_size=(32, 32), strides=(32, 32))(conv_7)
        output_layer = Activation('softmax')(deconv_11)

    model = Model(inputs=vgg.input, outputs=output_layer)
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


if __name__ == '__main__':

    train_folder='D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/fcn-train/train-fcn'
    valid_folder='D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/fcn-train/test-fcn'
    test_folder = 'D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/fcn-train/val-fcn'
    num_of_training_samples = len(os.listdir(train_folder))
    num_of_testing_samples = len(os.listdir(valid_folder))
    train_gen = DataGenerator(train_folder, batch_size=batch_size)
    val_gen = DataGenerator(valid_folder, batch_size=batch_size)
    test_gen = DataGenerator(test_folder,batch_size=batch_size)

    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(width, height, 3))
    #
    model = fcn(vgg, fcn8=True)

    adam = Adam(lr=0.001, decay=1e-06)
    # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=adam,metrics = [iou, dice_coef, precision, recall])
    model.load_weights('D:/Lancaster-MSC/Satellite-Image/code/fcn-1.h5')
    res = model.evaluate(test_gen, steps=10)
    # filepath = "best-model-vgg.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    #
    # history = model.fit_generator(train_gen, epochs=30, steps_per_epoch=num_of_training_samples//batch_size,
    #                        validation_data=val_gen, validation_steps=num_of_testing_samples//batch_size,
    #                        callbacks=callbacks_list, use_multiprocessing=False)
    #
    # save_path = 'D:/Lancaster-MSC/Satellite-Image/code/fcn-1.h5'
    # model.save(save_path)


    fcn_model = load_model('D:/Lancaster-MSC/Satellite-Image/code/fcn-1.h5')
    fcn_loss ,fcn_acc = fcn_model.evaluate(test_gen,steps=1)
    print('________________________')
    print('IOU:       |   {:.2f}  |'.format(res[1] * 100))
    print('Dice Coef: |   {:.2f}  |'.format(res[2] * 100))
    print('Precision: |   {:.2f}  |'.format(res[3] * 100))
    print('Recall:    |   {:.2f}  |'.format(res[4] * 100))
    print('F1-score:  |   {:.2f}  |', (2 * (res[4] * 100) * (res[3] * 100)) / ((res[4] * 100) + (res[3] * 100)))
    print('Accuracy:  |   {:.2f}  |'.format(fcn_acc * 100))
    print("Loss:      |   {:.2f}  |".format(fcn_loss * 100))
    print('________________________')
    # IOU:       | 99.75 |
    # Dice Coef: | 76.15 |
    # Precision: | 85.49 |
    # Recall:    | 85.49 |
    # F1-score:  | 85.49 |
    # Accuracy:  | 87.25 |
    # Loss:      | 45.98 |


    # print(unet_acc) # 0.8793
    #
    # loss = history.history["val_loss"]
    # acc = history.history["val_accuracy"] #accuracy
    #
    # plot.figure(figsize=(12, 6))
    # plot.subplot(211)
    # plot.title("Val. Loss")
    # plot.plot(loss)
    # plot.xlabel("Epoch")
    # plot.ylabel("Loss")
    #
    # plot.subplot(212)
    # plot.title("Val. Accuracy")
    # plot.plot(acc)
    # plot.xlabel("Epoch")
    # plot.ylabel("Accuracy")
    #
    # plot.tight_layout()
    # # plot.savefig("learn.png", dpi=150)
    # plot.show()

    # val_gen = DataGenerator(valid_folder)
    max_show = 2
    imgs, segs = next(test_gen)
    pred = model.predict(imgs)
    for i in range(max_show):
        _p = give_color_to_seg_img(np.argmax(pred[i], axis=-1))
        _s = give_color_to_seg_img(np.argmax(segs[i], axis=-1))

        predimg = cv2.addWeighted(imgs[i] / 255, 0.5, _p, 0.5, 0)
        trueimg = cv2.addWeighted(imgs[i] / 255, 0.5, _s, 0.5, 0)
        print(predimg.shape)
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.title("Prediction")
        plt.imshow(predimg)
        plt.axis("off")
        plt.subplot(122)
        plt.title("Original")
        plt.imshow(trueimg, plt.cm.binary_r)
        plt.axis("off")
        plt.tight_layout()
        # plot.savefig("pred_" + str(i) + ".png", dpi=150)
        plt.show()