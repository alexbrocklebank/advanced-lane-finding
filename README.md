**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/orig_undistorted.png "Undistorted"
[image2]: ./output_images/orig_pt.png "Perspective Transform"
[image3]: ./output_images/orig_combined.png "Binary Example"
[image4]: ./output_images/histogram.png "Line Points"
[image5]: ./output_images/hist_highlight.png "Line Highlight"
[image6]: ./output_images/output.png "Output"
[video1]: ./video_output/output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 11 - 36 in camera.py, using the calibrate() method in the Camera class.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I used a loop to process all the calibration images.  I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

### Pipeline

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I applied this distortion correction to the first frame of video using the `cv2.undistort()` function and obtained this result:

![Original Image vs. Undistorted Image][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 76 through 83 in `lanelines.py`).  These lines of code reference helper functions in frame.py, where I created a Frame class to manipulate a given frame.  Here's an example of my output for this step.  

![Original vs. Thresholded Binary Image][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 94 through 100 in the file `frame.py`.  This function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points inside the `lanelines.py` file like this:

```python
src = np.float32([[200,720],[1100,720],[595,450],[685,450]])
dst = np.float32([[300,720],[980,720],[300,0],[980,0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 200, 720      | 300, 720        |
| 1100, 720      | 980, 720      |
| 595, 450     | 300, 0      |
| 685, 450      | 980, 0        |


![Perspective Transform][image2]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Within `lane.py` in lines 122 - 124 I fit the lane line pixels to polynomials using numpy's polyfit() function.  Below is the resulting image of finding the lane-line pixels:

![Lane-Line Points][image4]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented steps 5 and 6 at the same time. I did this in lines 263 through 269 in my code in `lane.py` to notate in the unwarped output, but it also uses a function vehicle_offset() on lines 232 - 240 for the vehicle positioning. in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![Annotated and Unwarped Output][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./video_output/output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Finding lane lines using computer vision and non-linear functions is much more robust than the initial lane finding project.  I spent a large portion of the project within the line detection logic after the image has been fed through the binary filter pipeline.  I had a great deal averaging out the lines between frames, which wound up being helped a great deal by modifying the parameters and colors used within the pipeline instead.  

Once I increased the thresholds on a few of the pipeline "layers" and increased the width and decreased the length of the perspective transform area, I managed to clean up the lane detected dramatically.  Where this pipeline still under-performs is in areas of concrete roads where the lines are washed out by the lighter color backdrop.  For this I need to tweak the pipeline some more and possibly add some more color layers to the binary logic.

A few things that will increase the robustness would be averaging the curve and lane lines between a few of the most recent frames, similar to the learning rate in neural networks.  Preventing drastic changes by a dampener such as this will improve accuracy and help when the road material changes or shadows appear.
