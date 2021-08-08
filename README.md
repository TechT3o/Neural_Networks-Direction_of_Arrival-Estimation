# Neural_Networks-Direction_of_Arrival-Estimation

## Repository Overview
This repository contains the files used in my dissertation project on deep Direction of Arrival estimation using a microphone array. It contains the Python code that was used to generate simulated data, to preprocess the data and train the neural networks and to create a graphical user interface (GUI) for controlling the working prototype. It also contains the Arduino code that is used to control the stepper motor of the rotating platform used in the demonstration and MATLAB code that is used to generate the beamforming weights needed in the 60-degree implementation. Some extra supporting files such as Python code to perform audio streaming between the microphone array and a Windows laptop, some functions that are useful for embedded microphone array development and test data that can be downloaded to test the code are also included. The audio data used come from the AMI Corpus and can be downloaded here (https://groups.inf.ed.ac.uk/ami/download/). The Respeaker_Core_v2.pdf has all the necessary information on how to set up the Respeaker Core v2 microphone array (https://wiki.seeedstudio.com/ReSpeaker_Core_v2.0/).

## Project Overview
The scope of this project is to implement a real-time deep neural network based direction of arrival (DOA)
estimation algorithm, that can localize multiple sound sources, in an embedded computer. The symmetry
of a 6 microphone uniform circular array (UCA) was exploited by finding a reference microphone closest
to a source using beamforming and inputting the generalized cross correlation with phase transform
(GCC-PHAT) matrix, for all microphone pair combinations, in a multilayer perceptron (MLP) and a convolutional neural network (CNN) that
were trained to predict the DOA in 60 degrees. This way all 360 degrees are covered but the classification
task of the MLP is reduced to 60 classes relaxing the training data requirement.

A GUI was made that includes a main menu, a rotation controller page, a data collection page and the DOA estimation page. The main menu allows you to go to the other pages. The rotation controller page allows you to input the desired degree and sendsthe command to the arduino to rotate the automatic platform to that degree. The data collection page allows you to input the starting angle, the stopping angle,the angle resolution and the number of audio samples to record in every degree and proceeds to automatically record a .wav dataset and the .csv files with the GCC data used in neural network training. The DOA estimation page is a demonstration of the DOA algorithm with a graph that indicates the DOA, a scale that alters the energy threshold of the voice activity of detection, a choice of the 360 degree algorithm or the 60 degree and beamforming implementation and a tickbox that allows the inclusion of the MUSIC algorithm.

### Project Components
#### For testing the code
* Seeed ReSpeaker Core v2
* 64-bit computer
#### For the working prototype
##### Mechanical parts:
* 3D printed rotating platform (.stl files found here: https://www.thingiverse.com/thing:4915795)
* 1x Nema 17 stepper motor
* 1x M3 x 6 screw and M3 nut (Rotating paltform - Rotor)
* 4x M3 x 8 screws and M3 nuts (Stepper motor - Base)

##### Electronics:
* 1x Arduino Nano Every
* 1x TMC2130 stepper motor driver module
* 1x 12V 2A power supply adapter
* Jumper wires

## Demo


https://user-images.githubusercontent.com/87833804/128599055-5842be82-1926-488d-b4ee-9d84eb3137da.mp4 

