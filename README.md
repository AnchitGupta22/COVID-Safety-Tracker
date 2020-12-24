# COVID-Safety-Tracker
This project aims to track the two basic precautions as defined by the WHO - Social Distancing and Face Masks. 
The system uses DeepLearing and Machine Learning concepts like **object detection** and **convolutional neural networks** for its working.

![output](https://github.com/AnchitGupta22/COVID-Safety-Tracker/blob/main/assets/final_project_output.gif)

## Inspiration for the project

Link : https://www.aboutamazon.com/news/operations/amazon-introduces-distance-assistant

## Concepts used

1. Object detection using Yolo v3 to detect humans.

2. CNN for Face Mask detection. Current input size for the system is 48x48px.

## What is new in this project and how its different from the simple social distancing projects?

This project includes concepts like perspective transformation to eliminate almost all errors while performing **social distancing tracking**.
By using this concept, the depth factor of the video is eliminated and hence, the distance between each object is calculated more accurately.

At the same time, it also performs **mask detection** which adds to the system's capability to be of significant use. 

It also generates a simple dashboard depicting the stats of the above two mentioned norms. The stats include - **Total people**, **Violators** and **Non-violators**. The dashboard is created using user-defined functions using OpenCV functionalities.

## Input

The input can be either a pre-recorded or live video feed from a cctv camera. 

*Necessary changes in the code can be done accordingly*

## Flowchart 

Below is the flowchart for the entire system :-

![flowchart](https://github.com/AnchitGupta22/COVID-Safety-Tracker/blob/main/assets/flowchart_main.PNG)

## Working

1. The system reads the video feed using **OpenCV**.

2. On the first frame of the video feed, the system will ask you to do two things :
      
      (i)   Select a ROI for creating the Bird's Eye View of the scene. The number of points will be four.
      
      (ii)  Select a custom 6 feet distance for the scene, which can be the height of the tallest person in the frame or any other custom measure. The number of points will be 2.
      
      
3. Just after step 2, the system will generate a Bird's Eye View map of the selected ROI. It will also calculate a threshold distance on this view from step 2(ii), taking it as the required 6 feet distance.

4. From the second frame onwards, the system will perform three operations :-
    
      (i)     Perform **object-detection** using **Yolo v3** and filter out ```person``` class. Map these objects onto the Bird's Eye View. Calculate **Euclidian Distance** between these mapped points. Classify them as ```Violators``` and ```Non-violators``` accordingly. 
      
      (ii)    Perform **face approximation** [*more info below*](#face-approximation) on each detected person from step 4(i). Feed this approximated face into a function which will detect if the face has mask or not. Classify them as ```Violators``` and ```Non-violators``` accordingly. 
      
      (iii)   Generate the dashboard with two graphs, one for Social Distancing Tracking and another for Mask Detection. Use the stats from step 4(i) and ste 4(ii) to populate the two graphs.
      
## Face Approximation

Since the test video feed isn't of very good quality, and its recorded from a very great height, the faces in the feed aren't of a great size. The size was too small to perform **Face Detection** using either Yolo v3 or Haarcascading. Therefore, a temporary solution was to assume that our body consists of five equal parts, and that our face is in the top most part out of these five parts. Hence, accordingly our face will be in the top 20% of the bounding box generated in object detection.

Given below is an illustartion :-

![face approximation](https://github.com/AnchitGupta22/COVID-Safety-Tracker/blob/main/assets/face%20approximation.PNG)

For the current system, the value is taken to be 24% or 0.24.

## Metrics for the Face Mask Detection model

```Input size``` for the model is taken to be 48x48px. 

Accuracy Graph             |  Loss Graph
:-------------------------:|:-------------------------:
![](https://github.com/AnchitGupta22/COVID-Safety-Tracker/blob/main/assets/acc_graph.PNG)  |  ![](https://github.com/AnchitGupta22/COVID-Safety-Tracker/blob/main/assets/loss_graph.PNG)


## Application

This system can be installed in organisations to monitor employee safety and take necessary steps accordingly. For example, it can be installed in schools and colleges to ensure everyone is following the specified norms.

## Improvements

1. The current project setup uses only one camera which allows us to succesfully monitor social distancing. But mask detection is a little error prone due to the height of the camera and its distance from the ROI. A better setup would be to use two cameras where the lower one can be used to detect masks whereas the one above it at a much greater height can be used to track social distancing, as depicted in the below diagram.

![two camera setup](https://github.com/AnchitGupta22/COVID-Safety-Tracker/blob/main/assets/two%20camera%20setup.PNG)

2. The system presently is also running mask detection on the back-faced faces, which gives inaccurate results. This can be removed by adding another classifier to detect front-faced and back-faced faces and only pass front-faced faces through the mask detection model.

3. SRGANs can be used to improve the quality of the face extracted from the detected person.

4. Flask based interactive application can also be created.

## Project requirements

1. matplotlib==3.0.0

2. numpy==1.15.2

3. Keras==2.2.4

4. tensorflow==1.10.0

5. opencv_python==4.0.0.21

6. scikit_image==0.15.0

7. imageio==2.9.0

8. Pillow==8.0.1

9. skimage==0.0

## Contributions

Currently, the project is not open sourced. If you would like to contribute and imrove the current system, contact me at gupta.anchit.99@gmail.com
