# Analysis-for-100-lakes
* This project used satellite image data from Google Earth to analyse the changes in the area of over 100 lakes worldwide over the last 30 years (1991-2020) using deep neural networks.

* In the visualisation part, this project used Basemap to show the changes of each lake on a map and used the data obtained after image segmentation to show the overall changes of the lakes in different regions using curves. This result presents a very interesting phenomenon.

* The following guide will help you to install and run the project on your local machine for development and testing. For specific implementation steps, please refer to the deployment subsection.

## Dataset

### Data Collection
* Firstly, you need to Download Google Earth on you local machine, and then you can zoom in or out to select the lake area. Meanwhile, you can select a specific year of Satellite image to observe the variation of this lake. I have already uploaded 104 lakes in the Dataset file, you can use those images directly or select some new lakes in Google Earth.

### Data Pre-processing
* I chose 200 images as the trainning set for my Neural Network, 150 of which were the train set and the remaining 50 were the test set. To begin with, I used the k-means algorithm to segment the original image, labelling the lake area and the background area of the image using different colours, then using ColorMap to greyscale the image, and finally converting the image into a black and white segmented image by manual adjustment. 

![image](https://github.com/ChandlerGeng/Analysis-for-100-lakes/blob/main/data%26graph/Fig1.png)

## Neural Network Models
* Three deep neural network models (FCN, U-net and Segnet) are used in this project and detailed code is provided in the Neural Network Models file, which includes how to place images as input layers into the neural network and the performance scores for each model. Besides, 

* 

