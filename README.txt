Source code for paper: Su et al., Deep End-to-End Time-of-Flight Imaging, CVPR '18


File structure:

./src/GAN:
the main training and testing code derived from Zhu et al., CycleGAN, implemented in Torch. The README.txt file should provide sufficient information for running the training/testing code, and how to visualize the results, including the instructions on downloading a pretrained model. 

./src/pbrt-v3-tof: 
modified version of PBRT-v3 that implements time-resolved rendering. You may check and compare with the following commit of the official PBRT here to understand our implementation, https://github.com/mmp/pbrt-v3/tree/9f0c175cf1c58b36572526bea2d6902284bb2c36. When used in published research, please also cite O'Toole et al., "Reconstructing Transient Images from Single-Photon Sensors", CVPR 2017, who kindly provided the initial implementation of the renderer.

./script: 
the associated matlab scripts (messy) for converting transient images to the network inputs, and visualization, etc. If you need the raw Tintin data (http://www.ti.com/tool/OPT8241-CDK-EVM), please let me know. 

./pbrt-v3-scenes:
our modified version of the publicly available PBRT-v3 datasets (https://pbrt.org/scenes-v3.html), with modified light source, simplified material properties, and manually generated camera trajectories,. See ./script for code that renders and processes the .pbrt scene files. 


For any further questions send me an email. 


Copyright (C) 2018. Shuochen Su
Email: shuochsu@cs.ubc.ca

