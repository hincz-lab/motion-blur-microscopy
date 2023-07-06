# motion-blur-microscopy

**This README file is currently under construction.

## Table of Contents
1. [Introduction](##introduction)
2. [Training Code/Data navigation](##training-codedata-navigation)
   * [Phase 1 Code/Data](###phase1codedata)
   * [Phase 2 Code/Data](###phase2codedata)
3. 

## Introduction

Hello, and welcome to the Github page which houses the code and data used for the paper Motion Blur Microscopy by Goreke, Gonzales, Shipley, et al. In this Github repository, you will find a complete collection of code used and data used/generated in the production of results for the paper. In this paper, we developed a machine learning protocol where adhered cells could be analysed in the Motion Blue Microscopy framework. The relevant code and data can be largely split into two categories.

1. Training Code/Data
2. Analysis Code/Data

Code and data for the two larger categories can be found in the [Training_Material](/Training_Material/) and [Analysis_Material](/Analysis_Material) directories respectively. The following sections will guide readers on how to navigate both categories.

## Training Code/Data Navigation

When you enter the [Training_Material](/Training_Material/) directory, you will notice many sub-directories. The idea here is that we want to decompose the training process into pieces that are more accessible to the community. In each sub-directory, there will be a corresponding Jupyter notebook, as well as sub-sub-directories, which contain inputs to the code and outputs from the code that were used and produced in our work. The "official" order of sub-directories to be followed (with descriptions) is as follows:

### Phase 1 Code/Data

Phase 1 of the machine learning workflow is a semantic segmnatation network, whose job is to classify every pixel of an input motion blue microscopy into one of two categories, background, or adhered.

* [Extract Video Frames](/Training_Material/Extract_Video_Frames/)
  The code in this sub-directory is a helper code, which can extract individual frames from a motion blur microscopy video. This process is useful if readers want to use frames from a video as training images for the phase one network.
  - Input:
    A motion blur microscopy video.
  - Output:
    Individual frames from the input video.

* [Complete Mask Coloring](/Training_Material/Complete_Mask_Coloring/)
  As a starting point, a training image(s), or frame(s) from a training video, should be manually colored. The user should color over all of the "adhered" regions of the training images, and leave the rest of the image untouched. We personally labeled our images using the software [Gimp](https://www.gimp.org/). The code in this sub-directory will take an image, or images labeled in this way and fill all non-colored regions as background, which is noted by the color black [0,0,0] (RGB).
  - Input:
    Partially colored masks, where the "adhered" regions of the original images are colored.
  - Output:
    A fully colored mask, where the "adhered" regions of the original image are colored the same as the input, and the rest of the pixels in the image are colored in black for "background".

* [Label And Layer Masks](/Training_Material/Label_And_Layer_Masks/)
  The code in this sub-directory will convert the completed colored masks into integer labeled regions. On top of this, the code will convert the labeled regions into layered one-hot labeled regions.

  - Input:
    Fully colored masks.
  - Output:
    Integer labeled masks by category. As an example, 0 = Background, 1 = Adhered. Also, layered one-hot labeled masks.

* [Split Into Tiles](/Training_Material/Split_Into_Tiles/)
  The phase one network takes in as an input regions of a specific size, specifically, 128x128 pixels. The code here generates these sized tiles from our input images/masks/one-hot labels by first splitting each image into 150x150 pixel size chunks, and then rescaling to 128x128.

* [Train Phase One](/Training_Material/Train_Phase_One)


### Phase 2 Code/Data

* [Extract Phase Two Regions](/Training_Material/Extract_Phase_Two_Regions/)


* [Create Train Split](/Training_Material/Create_Train_Split_Phase_Two/)

* [Train Phase Two](/Training_Material/Train_Phase_Two/)


## Analysis Code/Data Navigation
