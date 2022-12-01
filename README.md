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
* Three deep neural network models (FCN, U-net and Segnet) are used in this project and detailed code is provided in the Neural Network Models file, which includes how to place images as input layers into the neural network and the performance scores for each model. In addition I compared the performance scores of these models, with U-net having the best overall performance.

![image](https://github.com/ChandlerGeng/Analysis-for-100-lakes/blob/main/data%26graph/model-performance.png)

* Finally I used the idea of Ensemble Learning to optimise the predictions of several neural network models using a Weighted Voting method to obtain a more accurate segmentation graph.

![image](https://github.com/ChandlerGeng/Analysis-for-100-lakes/blob/main/data%26graph/ensemble-new.png)

## Findings and Discussion
* By using the above method I have derived data on the change in area of each lake over a 30 year period and used the rate of change in area to analyse the growth and shrinkage of the lake and finally displayed this change on a global map using the BaseMap library in Python. The rate of change is shown in blue if it is ±5%, increasing yellow if it is greater than 5% (the more it grows the brighter the yellow), and increasing red if it is less than 5%.

![image](https://github.com/ChandlerGeng/Analysis-for-100-lakes/blob/main/data%26graph/world_variation.png)

* The above graph shows that the lake area in the Tibetan Plateau region has generally shown an increasing trend due to the effects of glacial melt and increased rainfall. Between 1991 and 1997 there was a slight decrease in lake area due to the delayed permafrost effect. The graph below shows the change in area of the three lakes in the region.

![image](https://github.com/ChandlerGeng/Analysis-for-100-lakes/blob/main/data%26graph/Himalayas.png)

* Another noticeable phenomenon is the severe reduction in the size of lakes in Central Asia, especially in southern Afghanistan, which have almost disappeared in the last 10 years.

* From a global perspective, there is a declining trend in the overall size of the lake. Even more interesting is the fact that the area of the lake fluctuates almost every three years. The reasons for this phenomenon have yet to be analysed and demonstrated.

![image](https://github.com/ChandlerGeng/Analysis-for-100-lakes/blob/main/data%26graph/figure4.png)
