# Motion Detector

## The problem

* Detect moving objects in a video stream by using only signal processing techniques
* Avoid using machine learning techniques

## The solution

  Build a pipeline consisting of background removal techniques and Fast Fourier Transform:
  
   * Take every frame from the video and apply background subtraction
   * Preprocess every frame by resizing and applying a Gaussian Blur filter
   * Convert the frames to grayscale
   * Compute 3D Fast Fourier Transform on the whole sequence of frames
   * Compute the phase angle
   * Compute phase spectrum array from the phase angle
   * Apply 3D inverse Fast Fourier Transform on the phase spectrum array
   * For every element in the array apply a Gaussian Blur filter in order to reduce the noise and convert it to a binary image using the mean as threshold
   * Find contours around the resulting white regions and eliminate the areas that are too small
   
## Advantages

* This approach views the stream as a whole and acts accordingly
* In the case of using only background subtraction techniques these techniques are applied on individual frames only

## Downsides

* Uses lots of hardware resources (memory and CPU)
* It cannot differentiate between actual moving objects and apparent moving objects (like water surface or foliage)

## What to improve

* Filter better the movement detection in order to distinguish between an actual moving object and foliage or moving water surface

## Useful links (instead of bibliography):

* https://dsp.stackexchange.com/questions/16462/how-moving-part-pixel-intensity-values-of-video-frames-becomes-dominant-compared
* https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
* https://docs.opencv.org/3.2.0/d3/db0/tutorial_py_table_of_contents_video.html
* https://dsp.stackexchange.com/questions/23758/is-it-possible-to-do-single-vehicle-tracking-using-fourier-transform
* https://www.youtube.com/watch?v=bSeFrPrqZ2A
* https://www.gaussianwaves.com/2015/11/interpreting-fft-results-obtaining-magnitude-and-phase-information/
