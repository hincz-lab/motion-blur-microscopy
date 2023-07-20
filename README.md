# motion-blur-microscopy

**This README file is currently under construction.

## Table of Contents
1. [Introduction](#introduction)
2. [Training Code/Data Navigation](#training-codedata-navigation)
   * [Phase 1](#phase-1-codedata)
   * [Phase 2](#phase-2-codedata)
3. [Analysis Code/Data Navigation](#analysis-codedata-navigation)

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

  This code was last run without errors with the following library versions:

  - python 3.9.15
  - matplotlib 3.6.2
  - numpy 1.23.4

* [Label And Layer Masks](/Training_Material/Label_And_Layer_Masks/)
  The code in this sub-directory will convert the completed colored masks into label encoded regions. On top of this, the code will convert the label encoded regions into layered one-hot encoded regions.

  - Input:
    Fully colored masks.
  - Output:
     Label encoded masks. Also, layered one-hot encoded masks.

  This code was last run without errors with the following library versions:

  - python 3.9.15
  - matplotlib 3.6.2
  - numpy 1.23.4
  - tensorflow 2.10.0
  - keras 2.10.0

* [Split Into Tiles](/Training_Material/Split_Into_Tiles/)
  The phase one network takes in as an input regions of a specific size, specifically, 128x128 pixels. The code here generates these sized tiles from our input images/colored masks/label encoded masks/one-hot encoded masks by first splitting each image into 150x150 pixel size chunks, and then rescaling to 128x128.

- Input:
  Original images, colored masks, label encoded masks, and one-hot encoded masks.
- Output:
  All of the input images/masks split into 150x150 pixel tiles and 128x128 pixel tiles.

  This code was last run without errors with the following library versions:

  - python 3.9.15
  - matplotlib 3.6.2
  - numpy 1.23.4
  - opencv 4.6.0

* [Train Phase One](/Training_Material/Train_Phase_One)
  Here we actually train the phase one segmentation network. The network architecture is inspired by U-Net, and the Hinczewski Lab's previous work.

  - Input:
    Original image tiles, as well as one-hot encoded mask tiles of size 128x128 pixels.
  - Output:
    A trained network, as a .h5 file, which includes the network architecture and the associated weights, all in one.

  This code was last run without errors with the following library versions:

  - python 3.9.15
  - matplotlib 3.6.2
  - numpy 1.23.4
  - opencv 4.6.0
  - tensorflow 2.10.0
  - keras 2.10.0
  - pandas 1.5.2

### Phase 2 Code/Data

* [Extract Phase Two Regions](/Training_Material/Extract_Phase_Two_Regions/)
  In this sub-directory, you will find code which takes regions identified by the phase one segmentation network as adhered, and extracts a 40x40 pixel square centered on the adhered region. These regions will need to be manually classified by cell type by the user for use in the phase 2 network training. The purpose of this code is to speed up the process of identifying regions from the phase one network to be manually classified. In our analysis, we rescale the color of the inputs.

  - Input:
      Raw MBM images or frames from MBM videos.
  - Output:
      40x40 pixel regions corresponding to areas of the raw images or frames classified as "adhered" by the phase one network.

  This code was last run without errors with the following library versions:

  - python 3.9.15
  - matplotlib 3.6.2
  - numpy 1.23.4
  - opencv 4.6.0
  - tensorflow 2.10.0
  - scikit-image 0.18.1
  - scipy 1.9.3

* [Create Train Split](/Training_Material/Create_Train_Split_Phase_Two/) In this sub-directory, you will find code which splits manually labeled cells into a training set, validation set, and testing set. The splits can be adjusted as the user wants.

  - Input:
     Manually classified images of regions identified as "adshered" by the phase one network of size 40x40 pixels.
    
  - Output:
    The input images will be split into a training set, validation set, and testing set, used for training the phase 2 network.

  This code was last run without errors with the following library versions.

  - python 3.9.15
  - matplotlib 3.6.2
  - numpy 1.23.4

* [Train Phase Two](/Training_Material/Train_Phase_Two/) In this sub-directory, you will find code which will use transfer learning to train a VGG16 network architecture with weights pre-trained on imagenet to classify cell types from one another.

  - Input:
    Manually classified regions identified from the phase one network split into training and validation sets.

  - Output:
    A trained VGG16 network, which can be used to classify adhered regions identified by the phase one segmantation network.

  This code was last run without errors with the following library versions.

  - python 3.9.15
  - tensorflow 2.10.0
  - keras 2.10.0

## Analysis Code/Data Navigation
When you enter the [Analysis Material](/Analysis_Material/) directory, you will notice many subdirectories. The idea here, just as with the training material directory, is to decompose all of the analysis code and data into smaller chunks more easily understandeable for a reader. In each subdirectory, you will notice a Jupyter notebook script, as well as sub-subdirectories, which contain inputs and outputs that we used/generated in our work. The official "order" to run the code in is as follows:

### Results Generation
The code in this section of the readme is used to take raw inputs, with the trained machine learning networks, and generate data from them. These results might be counts of cells, or morphological features for static images, or dynamic quantities in the case of videos.

* [Extract Morphological Features](/Analysis_Material/Extract_Morphological_Features/). This sub-directory will have two relevant scripts. One of the scripts can be used to extract morphological features (size, eccentricity) of regions classified as "adhered" by the phase 1 network, whereas the second script can be used to extract morphological features (size, eccentricity) of regions classified as "adhered" by the phase 1 network where input images are color adjusted.

  - Input:
    Raw MBM images or MBM video frames.

  - Output:
    Two .npy numpy arrays containing all of the region sizes and eccentricities.

    These codes were last run without errors with the following library versions:

    - python 3.9.15
    - matplotlib 3.6.2
    - numpy 1.23.4
    - opencv 4.6.0
    - tensorflow 2.10.0
    - scipy 1.9.3
    - scikit-image 0.18.1

* [Count Cells](/Analysis_Material/Count_Cells/) This sub-directory will have two relevant scripts. One of the scripts can be used to count cells for input MBM images or MBM frames using a size threshold. The second script can be used to count cells for input MBM images or MBM frames using a phase 2 classification network.

  - Input:
    Raw MBM images or MBM video frames.
  
  - Output:
    A .csv file which contains counts of relevant cells for all of the input images.

    These codes were last run without errors with the following library versions:

    - python 3.9.15
    - matplotlib 3.6.2
    - numpy 1.23.4
    - opencv 4.6.0
    - tensorflow 2.10.0
    - keras 2.10.0
    - scipy 1.9.3
    - scikit-image 0.18.1
    - pandas 1.5.2
   
* [Complete F1 Analysis](/Analysis_Material/Create_F1_Plot/) The code in this sub-directory can be used to complete an F1 analysis for the phase two classification network.

  - Input: The input will be "adhered" regions identified by the phase one segmentation network that were NOT used in the training or validation of the phase 2 classification network.
  
  - Output: The output will be three .npy numpy arrays containing the precision, recall, and F1 score values for a range of confidence thresholds.
 
    This code was last run without errors with the following library versions:

    - python 3.9.15
    - matplotlib 3.6.2
    - numpy 1.23.4
    - opencv 4.6.0
    - tensorflow 2.10.0

* [Video Analysis](/Analysis_Material/Video_Analysis/)

  - Input:
 
  - Output:


### Results Analysis
The code in this section is used to take data generated from raw inputs and create plots, tables, or any other sort of important representation of the results for the paper.

* [Create Hexplots](/Analysis_Material/Create_Hexplots/) This sub-directory will have code which will convert input region areas and eccentricities into a hexplot.

  - Input: Areas and eccentricities of regions identified by the phase 1 segmentation network as "adhered".
  
  - Output: A hexplot of the region areas and eccentricities.
 
    This code was last run without errors with the following library versions:

    - python 3.9.15
    - matplotlib 3.6.2
    - numpy 1.23.4
    - pandas 1.5.2

* [Create Reproducibility Plots](/Analysis_Material/Create_Reproducibility_Plots/) The code in this sub-directory can create inter- and intra-reproducibility plots.

  - Input: A .csv file containing counts generated from two different experimenters at different times of an MBM experiment.
  
  - Output: Two plots, one for inter-reproducibility, and another for intra-reproducibility.
 
    This code was last run without errors with the following library versions:

    - python 3.9.15
    - matplotlib 3.6.2
    - numpy 1.23.4
    - pandas 1.5.2

* [Create R Squared Plots](/Analysis_Material/Create_R_Squared_Plots/)

  - Input:
  
  - Output:

* [Create R Squared Plots With Groupings](/Analysis_Material/Create_R_Squared_Plots_With_Groupings/)

  - Input:
  
  - Output:

* [Create Adhesion Time Probability Plots](/Analysis_Material/Create_Adhesion_Time_Probability_Plots/)

  - Input:
  
  - Output:

* [Create Eccentricity Vs Adhesion Time Plot](/Analysis_Material/Create_Eccentricity_Vs_Adhesion_Time_Plot/)

  - Input:
  
  - Output:

* [Create Average Velocity Plots](/Analysis_Material/Create_Average_Velocity_Plots/)

  - Input:
  
  - Output:

* [Create F1 Plot](/Analysis_Material/Create_F1_Plot/)

  - Input:
  
  - Output:
