import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from skimage.color import rgb2hsv
import os
import cv2
import matplotlib
from glob import glob
lakes = sorted(glob("D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/train-total/*.jpg"))
# save_path2 = "D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/train2/"
#
# for i in range(200):
#     I = Image.open(lakes[i])
#
#     save_out2 = save_path2 + str(i+1) + ".jpg"
#     I.save(save_out2)
#
# X = []
kmeans = KMeans(random_state=0, init='random', n_clusters=3)
save_path = "D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/seg-clu3/"

name = 1
for i in zip(lakes):
    img = Image.open(i[0])

    z = (np.dstack((np.array(img), rgb2hsv(np.array(img)))))

    vectorized = np.float32(z.reshape((-1, 6)))
    #X.append(vectorized)
    labels = kmeans.fit_predict(vectorized)
    pic = labels.reshape(1080, 1920)
    save_out = save_path + str(name) + ".jpg"
    matplotlib.image.imsave(save_out, pic)
    name = name + 1

open_path = "D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/seg-clu3/"
save_path = "D:/Lancaster-MSC/Satellite-Image/suppervised-train-200/seg-clu8/"
for i in range(200):
    x = i + 1
    open_path1 = open_path + str(x) + ".jpg"
    save_path2 = save_path + str(x) + ".jpg"
    img_gray = cv2.imread(open_path1,cv2.IMREAD_GRAYSCALE)
    dst = cv2.applyColorMap(img_gray, 1)
    cv2.imwrite(save_path2, dst)

