import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
# Local Project Libraries:
from camera import Camera
import line
from line import Line
from frame import Frame


# Variables
calibrate_dir = "camera_cal/"
chessboard = (9,6)
test_img = mpimg.imread("test_images/straight_lines2.jpg")
#test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2HSV)
# Sobel Kernel Size
ksize = 3
# 4 Source points in Lane Trapezoid
src = np.float32([[267,675],[608,442],[679,442],[1053,675]])
# 4 Destination points dst = np.float32([[,],[,],[,],[,]])
dst = np.float32([[275,719],[275,0],[1020,0],[1020,719]])

# Objects
cam = Camera()
lane_left = Line()
lane_right = Line()


# Display Test Output
def test(image1, title1, image2, title2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(title1, fontsize=50)
    ax2.imshow(image2, cmap='gray')
    ax2.set_title(title2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


# VIDEO
#output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
#clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#clip.write_videofile(output, audio=False)


# Step 1: Calibrate Camera
cam.calibrate(calibrate_dir, chessboard)
undist = cam.undistort(test_img)

# Step 2: Set up Frame pipeline
frame = Frame(test_img, undist)
print("Height: {}, Width: {}".format(frame.height, frame.width))

ch = frame.HLS[:,:,2]
gradx = frame.abs_sobel_thresh(ch ,orient='x', sobel_kernel=ksize, thresh=(30, 120))
grady = frame.abs_sobel_thresh(ch ,orient='y', sobel_kernel=ksize, thresh=(30, 120))
mag_binary = frame.mag_thresh(ch ,sobel_kernel=ksize, mag_thresh=(30, 120))
dir_binary = frame.dir_threshold(ch ,sobel_kernel=ksize, thresh=(0.7, 1.3))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
test(frame.image, "Original Frame", combined, "Combined Image")

# Step 3: Perspective Transform
p_t = frame.perspective_transform(combined, src, dst)
test(frame.image, "Original Frame", p_t, "Perspective Transform")

# Step 4: Lane Lines
line.find_lines(lane_left, lane_right, p_t, test_img)
line.draw(lane_left, lane_right, frame)
