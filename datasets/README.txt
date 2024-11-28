
Fire - v3 2023-11-26 12:42am

JeonbukNational University

Linfeng Wang ----》 wlf15130930911@gmail.com
Oualid Doukhi ----> doukhioualid@gmail.com
==============================

https://github.com/WangLF1996/FCDNet-Dataset-and-Algorithm

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 3078 images.
Fire are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -15 and +15 degrees
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically


