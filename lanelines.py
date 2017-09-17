import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
# Local Project Libraries:
from camera import Camera
import line
from line import Line
from lane import Lane
from frame import Frame


# TODO: Allow command line switches
# Variables
calibrate_dir = "camera_cal/"
chessboard = (9,6)
test_img = mpimg.imread("test_images/test5.jpg")
#test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2HSV)
# Sobel Kernel Size
ksize = 3
# 4 Source points in Lane Trapezoid
#src = np.float32([[267,675],[608,442],[679,442],[1053,675]])
src = np.float32([[200,720],[1100,720],[595,450],[685,450]])
# 4 Destination points dst = np.float32([[,],[,],[,],[,]])
#dst = np.float32([[275,719],[275,0],[1020,0],[1020,719]])
dst = np.float32([[300,720],[980,720],[300,0],[980,0]])
output_video = "video_output/output.mp4"
input_video = 'project_video.mp4'

# Objects
cam = Camera()
lane_left = Line()
lane_right = Line()
lane = Lane()


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


# Step 1: Calibrate Camera
cam.calibrate(calibrate_dir, chessboard)


def process_image(image):
    undist = cam.undistort(image)

    # Step 2: Set up Frame pipeline
    frame = Frame(image, undist)
    #print("Height: {}, Width: {}".format(frame.height, frame.width))
    #test(frame.image, "Original Frame", undist, "Undistorted Image")

    S = frame.HLS[:,:,2]
    hLs = frame.HLS[:,:,1]
    Luv = frame.LUV[:,:,0]
    B = frame.LAB[:,:,2]
    gray = frame.gray

    '''gradx = frame.abs_sobel_thresh(L ,orient='x', sobel_kernel=ksize, thresh=(225, 255))
    test(frame.image, "Original Frame", gradx, "LUV L Binary Sobel X")# blank
    grady = frame.abs_sobel_thresh(L ,orient='y', sobel_kernel=ksize, thresh=(225, 255))
    test(frame.image, "Original Frame", grady, "LUV L Binary Sobel Y")# blank
    mag_binary = frame.mag_thresh(L ,sobel_kernel=ksize, mag_thresh=(225, 255))
    test(frame.image, "Original Frame", mag_binary, "LUV L Binary Mag Thresh") # blank
    dir_binary = frame.dir_threshold(L ,sobel_kernel=ksize, thresh=(0.7, 1.3))
    test(frame.image, "Original Frame", dir_binary, "LUV L Dir Binary") # horrible noise
    hls_bin = frame.hls_thresh(image, thresh=(170, 255))
    test(frame.image, "Original Frame", hls_bin, "LUV L HLS Binary") #Dark shadows amplified, lines in shadow hidden
    combined = np.zeros_like(dir_binary)
    combined[(gradx == 1 | ((mag_binary == 1) & (dir_binary == 1))) | hls_bin == 1] = 1
    test(frame.image, "Original Frame", combined, "LUV Combined L Binary") # same as hls_bin

    gradx = frame.abs_sobel_thresh(B ,orient='x', sobel_kernel=ksize, thresh=(155, 200))
    test(frame.image, "Original Frame", gradx, "LAB B Binary Sobel X") # blank
    grady = frame.abs_sobel_thresh(B ,orient='y', sobel_kernel=ksize, thresh=(155, 200))
    test(frame.image, "Original Frame", grady, "LAB B Binary Sobel Y") # blank
    mag_binary = frame.mag_thresh(B ,sobel_kernel=ksize, mag_thresh=(155, 200))
    test(frame.image, "Original Frame", mag_binary, "LAB B Binary Mag Thresh") # almost blank
    dir_binary = frame.dir_threshold(B ,sobel_kernel=ksize, thresh=(0.7, 1.3))
    test(frame.image, "Original Frame", dir_binary, "LAB B Dir Binary") # noisy, yellow line on concrete shows
    hls_bin = frame.hls_thresh(image, thresh=(170, 255))
    test(frame.image, "Original Frame", hls_bin, "LAB B HLS Binary") # pretty good
    combined = np.zeros_like(dir_binary)
    combined[(gradx == 1 | ((mag_binary == 1) & (dir_binary == 1))) | hls_bin == 1] = 1
    test(frame.image, "Original Frame", combined, "LAB Combined B Binary") # same as hls_bin

    gradx = frame.abs_sobel_thresh(S ,orient='x', sobel_kernel=ksize, thresh=(30, 170))
    #test(frame.image, "Original Frame", gradx, "HLS S Binary Sobel X") # pretty full image, white lines better
    grady = frame.abs_sobel_thresh(S ,orient='y', sobel_kernel=ksize, thresh=(30, 170))
    #test(frame.image, "Original Frame", grady, "HLS S Binary Sobel Y") # Also pretty good
    mag_binary = frame.mag_thresh(S ,sobel_kernel=ksize, mag_thresh=(30, 170))
    #test(frame.image, "Original Frame", mag_binary, "HLS S Binary Mag Thresh") # pretty good
    dir_binary = frame.dir_threshold(S ,sobel_kernel=ksize, thresh=(0.7, 1.3))
    #test(frame.image, "Original Frame", dir_binary, "HLS S Dir Binary") # noisy as hell
    hls_bin = frame.hls_thresh(image, thresh=(170, 255))
    #test(frame.image, "Original Frame", hls_bin, "HLS S HLS Binary") # Shadow affected
    combined = np.zeros_like(dir_binary)
    combined[(gradx == 1 | ((mag_binary == 1) & (dir_binary == 1))) | hls_bin == 1] = 1
    #test(frame.image, "Original Frame", combined, "HLS Combined S Binary") # Good'''

    sobel_x_binary = frame.abs_sobel_thresh(hLs, orient='x', sobel_kernel=ksize,
                                   thresh=(20, 255))
    #test(frame.image, "Original Frame", sobel_x_binary, "Sobel X Binary")

    s_binary = np.zeros_like(S)
    s_binary[(S >= 120) & (S <= 255)] = 1
    #test(frame.image, "Original Frame", s_binary, "S Binary")

    l_binary = np.zeros_like(hLs)
    l_binary[(hLs >= 40) & (hLs <= 255)] = 1
    #test(frame.image, "Original Frame", l_binary, "L Binary")

    combined = 255*np.dstack((l_binary, sobel_x_binary, s_binary)).astype('uint8')
    #test(frame.image, "Original Frame", combined, "Combined")
    binary = np.zeros_like(sobel_x_binary)
    binary[((l_binary == 1) & (s_binary == 1) | (sobel_x_binary == 1))] = 1
    binary = 255*np.dstack((binary, binary, binary)).astype('uint8')
    #test(frame.image, "Original Frame", binary, "New Binary")

    # Step 3: Perspective Transform
    p_t = frame.perspective_transform(binary, src, dst)
    #test(frame.image, "Original Frame", p_t, "Perspective Transform")

    # Step 4: Lane Lines
    #if lane.find_lines(lane_left, lane_right, p_t, image, vis=True):
    if lane.find_lines(lane_left, lane_right, p_t, image):
        return lane.draw(lane_left, lane_right, frame)
    else:
        return frame.undist

#out = process_image(test_img)
#test(test_img, "In", out, "Out")
clip = VideoFileClip(input_video)
out_clip = clip.fl_image(process_image)
out_clip.write_videofile(output_video, audio=False)
