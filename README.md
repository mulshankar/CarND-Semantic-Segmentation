# CarND-Semantic Segmentation Project - Advanced Deep Learning
Self-Driving Car Engineer Nanodegree Program

---

[//]: # (Image References)
[image1]: ./Images/SceneUnderstandingSample.png
[image2]: ./Images/WhyFCNs.png
[image3]: ./Images/FCN.png
[image4]: ./Images/Skip.png
[image5]: ./Images/FCN8.PNG
[image6]: ./Images/um_000000.png
[image7]: ./Images/umm_000072.png
[image8]: ./Images/uu_000099.png

## Introduction

Traditional computer vision techniques like bounding box networks - YOLO and Single Shot Detectors are helpful from a classification perspective. Semantic segmentation goes beyond these traditional techniques and identifies information at pixel-level granularity. This significantly improves decision-making ability. Shown below is a sample image from NVIDIA of a semantic segmentation implementation for scene understanding. As seen, every pixel is classified into its corresponding class - road, pedestrian, cars, train etc... 

![alt text][image1]

## Project Objectives

The primary objective of this project is to build and train a fully convolutional network that performs semantic segmentation for a self driving car application. The goal is to identify road pixels in a given image. 

## Fully Convolutional Networks (FCNs)

Conventional deep neural networks consists of a sequence of convolutional layers followed by fully connected layers. This works great for classification type problems - for example, is this an image of a hot dog or not? 

![alt text][image2]

If the question is posed slightly different - where in the picture is the hot dog? This is a much more challenging problem that requires spatial information. This is where fully convolutional networks excel. They preserve spatial information and works with input images of varying sizes. FCNs have two primary pieces - encoder and decoder. The encoder part goes through feature extraction via convolutional layers. The decoder part upscales the output of the convolutional layers to match the input image size. This is achieved via de-convolution. 

![alt text][image3]

In addition to just convolution and de-convolution, skip connections are another useful mechanism in FCN implementations. Skip connections help accuracy of decoder by introducing information from prior layers of the encoder. Below is a sample of semantic segmentation with and without skip connections. 

![alt text][image4]

## Implementation

This work is inspired by the FCN-8 architecture presented in this UC Berkeley paper: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

![alt text][image5]

It is common for encoder layers to use popular networks like ImageNet, ResNet and VGG. These networks are popular for their accuracy in feature extraction from images. The encoder portion of the FCN-8 architecture is derived from the VGG16 network referenced here: https://arxiv.org/pdf/1409.1556.pdf . This Oxford’s model won the 2013 ImageNet competition with 92.7% accuracy. 

* The FCN is implemented using TensorFlow library in Python 3, along with other dependencies such as Numpy and Scipy

* The Kitti road data set was used for training and testing purposes

* The first step of the encoder implementation was to download the VGG model and extract layers 3, 4 and 7.

```
def load_vgg(sess, vgg_path):

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, ["vgg16"], vgg_path)
    
    graph = tf.get_default_graph()
    vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor

```
* A quick sanity check on layer size could be performed by tf.shape() function. Layer 3 has depth of 256, layer 4 with depth of 512 and layer 7 is the fully connected layer of 4096 nodes. Width and height is assigned based on image input size.

* The fully connected layer 7 to a 1x1 convolution. This completes the encoder part of the FCN-8 architecture.

```
	# 1x1 convolution of vgg layer 7
	layer7a_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 
								   padding= 'same', 
								   kernel_initializer= tf.random_normal_initializer(stddev=0.01),
								   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
```

* The decoder part upsamples the previous layer by a factor of 2. This is done to match dimensions with layer 4 of the VGG network. 

```
	layer4a_in1 = tf.layers.conv2d_transpose(layer7a_out, num_classes, 4, 
                                             strides= (2, 2), 
                                             padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
```

* With dimensions matched to layer 4, a skip connection is perfomed by a 1x1 convolution of layer 4 and a simple add operation.

```
    # 1x1 convolution of vgg layer 4
    layer4a_in2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    # skip connection (element-wise addition)
    layer4a_out = tf.add(layer4a_in1, layer4a_in2)
```


* Further upsampling and skip connections are added to match the output image size to input image size

* The network was trained using the below parameters. All training was done on the AWS cloud with GPU instance. 

```
EPOCHS = 30
batch_size=16
learning_rate=0.0009
keep_prob=0.5
Optimizer=Adam
Loss function = cross entropy

```

## Closure

A fully convolutional network was implemented to perform semantic segmentation for a self driving application. Few sample results from testing the network are shown below. 

![alt text][image6]

![alt text][image7]

![alt text][image8]


