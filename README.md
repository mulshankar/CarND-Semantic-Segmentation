# CarND-Semantic Segmentation Project
Self-Driving Car Engineer Nanodegree Program

---

[//]: # (Image References)
[image1]: ./Images/SceneUnderstandingSample.png
[image2]: ./Images/WhyFCNs.png
[image3]: ./Images/FCN.png
[image4]: ./Images/Skip.png
[image5]: ./Images/Capture2.PNG
[image6]: ./Images/Capture.PNG
[image7]: ./Images/NIS.PNG
[image8]: ./Images/ChiSquare.PNG

## Introduction

This project was designed in collaboration with the NVIDIA Deep Learning Institute for the Udacity Self-Driving Car Nanodegree program. Traditional computer vision techniques like bounding box networks - YOLO and Single Shot Detectors are helpful from a classification perspective. Semantic segmentation goes beyond these traditional techniques and identifies information at pixel-level granularity. This significantly improves decision-making ability. Shown below is a sample image from NVIDIA of a semantic segmentation implementation for scene understanding. As seen, every pixel is classified into its corresponding class - road, pedestrian, cars, train etc... 

![alt text][image1]

## Project Objectives

The primary objective of this project is to build and train a fully convolutional network that performs semantic segmentation for a self driving car application. The goal is to identify road pixels in a given image. 

## Fully Convolutional Networks (FCNs)

Conventional deep neural networks consists of a sequence of convolutional layers followed by fully connected layers. This works great for classification type problems - for example, is this an image of a hot dog or not? 

![alt text][image2]

If the question is posed slightly different - where in the picture is the hot dog? This is a much more challenging problem that requires spatial information. This is where fully convolutional networks excel. They preserve spatial information and works with input images of varying sizes. FCNs have two primary pieces - encoder and decoder. The encoder part goes through feature extraction via convolutional layers. The decoder part upscales the output of the convolutional layers to match the input image size. This is achieved via de-convolution. 

![alt text][image3]

In addition to just convolution and de-convolution, skip connections are also used. Skip connections help upscaling by introducing information from the original input image. Below is a sample of detecting bikers with and without skip connections. 

![alt text][image4]

## Algorithm Overview

Before diving into the algorithm, it is important to introduce the concept of Frenet coordinates - s and d. Frenet coordinates simplify the car position from cartesian (x,y) coordinates to road coordinates. The 's' value is the distance along the direction of the road. The first waypoint has an s value of 0 because it is the starting point. The d vector has a magnitude of 1 and points perpendicular to the road in the direction of the right-hand side of the road. The d vector can be used to calculate lane positions. Frenet coordinates along with time can accurately capture the desired safe trajectory for the vehicle to navigate. 



While frenet coordinates are good for trajectory planning, the localization as well as motion control modules are in the global X,Y coordinates. Therefore, once the trajectory is planned in the frenet coordinate system, it is converted into XY coordinate system for the car's motion control module. The algorithm can be sub-sectioned into 2 parts - (1) First part decides on lane (2) Second part plans the trajectory for the desired lane

* The car initially starts in lane 1 ( 0- left, 1 -middle and 2-right lanes)

* The localization module reports the car's current x,y,s,d,yaw and speed information. 

* The first check is performed using information provided by the sensor fusion module. Information about cars in the current lane of travel is processed. A threshold of 30 m was set to identify if the test vehicle is too close to a vehicle right in front of it. 

```
for (int i=0;i<sensor_fusion.size();i++)	
{
	double other_car_d=sensor_fusion[i][6]; // get the 'd' coordinate of the car

	if (other_car_d<=4+(4*lane) && other_car_d>=(4*lane))	{ // assuming our car is in center of lane, check + - 2 m in given lane
		
		double other_car_vx=sensor_fusion[i][3];
		double other_car_vy=sensor_fusion[i][4];
		double other_car_speed=sqrt(other_car_vx*other_car_vx+other_car_vy*other_car_vy);
		
		double other_car_s=sensor_fusion[i][5];
		
		other_car_s=other_car_s+ prev_path_size*0.02*other_car_speed; // predict where the car will be at the end of its current planned path
		
		if ((other_car_s > car_s) && (other_car_s-car_s<30)) {					
			car_ahead=true;
		}					
	}			
}

```
* A simple switch case block decides on available options for lane changes

```
// decide on which lane you want to shift

switch (lane) {			
	case 0:
		lane_to_change=1;
		break;
	case 1:
		lane_to_change=0;
		break;
	case 2:
		lane_to_change=1;
		break;
	default:
		cout<<"problem in lane variable"<<endl;			
}	

```

* If there is car ahead in the lane that is too close, the first step is to reduce speed and then explore if a lane change maneuver is safe to perform. The check uses sensor fusion information of cars in the "desired" lane. A velocity constraint was also added to minimize jerk or acceleration. For example, the car should not perform lane change at 50 mph. 

* Once the lane is decided, the next step is to generate a safe trajectory for the car to traverse. While the trajectory is planned for a distance of 90 m, the base loop runs at 20 ms. The car probably travels for about the first few meters. The remaining points from the previous planned path is retained. 

* A set of anchor points or way-points are generated at wide-spaced intervals (30 m in our case) in the frenet coordinate system. A helper function 'getXY' converts these way-points into the cartesian coordinate X,Y system

* These new way-points are now appended to the untraversed points of the previous planned trajectory. This ensures a smooth continuous path. 

* The spline library was used to fit a polynomial through these points. Spline ensures the generated polynomial goes through every single way-point.

* A simple calculation is then done based on current velocity, loop time and distance to compute the desired x,y coordinates.

* The desired finer x,y coordinates are then transferred to the motion control module that executes the path by controlling pedal, brake and steer. 

![alt text][image5]

## Closure

A trajectory planning algorithm was implemented on a self driving car for navigating around the Udacity provided test track. The car top speed was around 49.5 mph. The car navigated very well within the test track without going off the curbs under all circumstances. Multiple laps yielded the same result thus verifying repeatability. Shown below are results from a couple of test runs.

![alt text][image6]

![alt text][image7]



