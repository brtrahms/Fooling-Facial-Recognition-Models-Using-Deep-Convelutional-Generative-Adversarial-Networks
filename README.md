# Fooling Facial Recognition Models Using Deep Convelutional Generative Advasarial Networks

Brandon Trahms
Github: brtrahms

## Intro
Facial Recognition software is becoming more and more prevelant in our daily lives from our phones to law enforcment. To help prevent the misuse of these softwares we should further our understanding of these algorithms so they don't remain as simple black boxs in which we don't know what is inside. This project is aiming to study low barrier to entry pre-trained Facial Recognition models and their accuarcy using a Deep Convelutional Generative Advasarial Network or DCGAN. The DCGAN will be used to generate images, learning over time to generate images that are thought to have a person's face by the network. We can then look through these positive results and determine how accurate these images are compared to a human observer.

### Language:
- python

### Dependencies
- tensorflow
- keras
- keras_vggface
- matplotlib
- numpy

### Facial Recognition Models:
- keras_vggface - ResNet50
- keras_vggface - SENet50

## Experiment Description
The Facial Detection Models will be loaded in and set up in our python enviornment. Our DCGAN will be then built using Tensorflow following the basic architecture of taking in a randomized normal seed and outputing a resulting image. These outputed images will be scored on the confidence percentage given by the recognition model for a certain identity we choose and fed back to the DCGAN for optimization. Once the GAN has generated a number of detections past a certain confidence threshold, we can look through and see if there are any false positives.

### Hardware

GPU: NIVIDIA GeForce GTX 1660 Ti with Max-Q Design

## Results

### ResNet50

### SENet50

## References

DCGAN:
 - https://www.tensorflow.org/tutorials/generative/dcgan

VGGFACE Models:
 - https://github.com/rcmalli/keras-vggface#projects--blog-posts
 - https://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf
 - https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/

