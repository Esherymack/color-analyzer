# color-analyzer
 color analysis project for CS496 special topics - data science
 
 Analyzes images for their most dominant colors via a k-means algorithm, graphs that, then approximates the most common color to a named color, then categorizes that color under a broader color family (using HSV to classify)

# to use
you need some libraries:
 * sklearn
 * matplotlib
 * numpy
 * opencv2
 * skimage
 * colorsys

Change the `image_data_directory` global var if you want to use a different directory for images
 
Don't try to use pngs, they don't behave nicely
 
The included dataset is my artwork.
 
* On that note, all included images are protected under a Creative Commons Attribution-Noncommercial-No Derivative Works 3.0 License, you can find out more about that [here](https://creativecommons.org/licenses/by-nc-nd/3.0/)
 
I don't know what I'm trying to accomplish with it but here you go
   
 
References: 
 
* https://towardsdatascience.com/color-identification-in-images-machine-learning-application-b26e770c4c71
* https://www.dataquest.io/blog/tutorial-colors-image-clustering-python/
* https://realpython.com/python-opencv-color-spaces/
