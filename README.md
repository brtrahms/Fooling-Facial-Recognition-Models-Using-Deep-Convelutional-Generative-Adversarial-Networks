# GAN-Facial-Recognition-Stress-Test

Brandon Trahms

Github: brtrahms

## Intro
Facial recognition software is becoming more and more prevelant in our daily lives from our phones to law enforcment. To help prevent the misuse of these softwares we should further our understanding of these algorithms so they don't remain as simple black boxs in which we don't know what is inside. This project is aiming to study various low barrier to entry pre-trained Facial Recognition models and their accuarcy using a Generative Advasarial Network or GAN. The GAN will be used to generate images, learning over time to generate images that are thought to have a face by the network. We can then look through these positive results and determine how accurate these images are compared to a human observer.

## Dependencies

### Language:
python

### Imports:
cv2

tensorflow

### Facial Detection Models:
OpenCV - HaarCascadeClassifier Face Detector 

OpenCV - Caffe Face Detector

## Experiment Description
These Facial Detection Models will be loaded in and set up in our python enviornment. Our GAN will be then built using Tensorflow following the basic architecture of taking in a randomized seed number and outputing a resulting image. These outputed images will be scored on the confidence percentage given by the detection model and fed back to the GAN for optimization. Once the GAN has generated a number of detections past a certain confidence threshold, we can look through and see if there are any false positives.
