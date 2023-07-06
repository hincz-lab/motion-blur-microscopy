# motion-blur-microscopy

**This README file is currently under construction.

Hello, and welcome to the Github page which houses the code and data used for the paper Motion Blur Microscopy by Goreke, Gonzales, Shipley, et al. In this Github repository, you will find a complete collection of code used and data used/generated in the production of results for the paper. In this paper, we developed a machine learning protocol where adhered cells could be analysed in the Motion Blue Microscopy framework. The relevant code and data can be largely split into two categories.

1. Training Code/Data
2. Analysis Code/Data

Code and data for the two larger categories can be found in the [Training_Material](/Training_Material/) and [Analysis_Material](/Analysis_Material) directories respectively. The following sections will guide readers on how to navigate both categories.

## Training Code/Data Navigation

When you enter the [Training_Material](/Training_Material/) directory, you will notice many sub-directories. The idea here is that we want to decompose the training process into pieces that are more accessible to the community. In each sub-directory, there will be a corresponding Jupyter notebook, as well as sub-sub-directories, which contain inputs to the code and outputs from the code that were used and produced in our work. The "official" order of sub-directories to be followed (with descriptions) is as follows:

* [Extract_Video_Frames](/Training_Material/Extract_Video_Frames/)

* [Complete_Mask_Coloring](/Training_Material/Complete_Mask_Coloring/)
  As a starting point, a training image(s), or frame(s) from a training video, should be manually colored. The user should color over all of the "adhered" regions of the training images, and leave the rest of the image untouched.

  The code in this sub-directory will take an image, or images labeled in this way and fill all non-colored regions as background, which is noted by the color black [0,0,0] (RGB).

* []


## Analysis Code/Data Navigation
