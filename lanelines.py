import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
# Local Project Libraries:
from camera import Camera
from line import Line
from lane import Lane
from frame import Frame


# TODO: Allow command line switches
# Variables
calibrate_dir = "camera_cal/"
chessboard = (9, 6)
test_img = mpimg.imread("test_images/test5.jpg")
# Sobel Kernel Size
ksize = 3
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


def process_image(image):
    # Undistort Image
    undist = cam.undistort(image)

    # Set up Frame pipeline
    frame = Frame(image, undist)
    # test(frame.image, "Original Frame", undist, "Undistorted Image")

    S = frame.HLS[:, :, 2]
    hLs = frame.HLS[:, :, 1]
    # Luv = frame.LUV[:, :, 0]
    # B = frame.LAB[:, :, 2]
    # gray = frame.gray

    # Absolute Value Sobel X Gradient using L-Channel in HLS
    sobel_x_binary = frame.abs_sobel_thresh(hLs, orient='x',
                                            sobel_kernel=ksize,
                                            thresh=(20, 255))
    # test(frame.image, "Original Frame", sobel_x_binary, "Sobel X Binary")

    # Binary Thresholded from S-Channel of HLS
    s_binary = np.zeros_like(S)
    s_binary[(S >= 120) & (S <= 255)] = 1
    # test(frame.image, "Original Frame", s_binary, "S Binary")

    # Thresholded Binary of L-Channel from HLS
    l_binary = np.zeros_like(hLs)
    l_binary[(hLs >= 40) & (hLs <= 255)] = 1
    # test(frame.image, "Original Frame", l_binary, "L Binary")

    # Combined Binary of previous 3 binary images
    combined = 255*np.dstack((l_binary, sobel_x_binary,
                              s_binary)).astype('uint8')
    # test(frame.image, "Original Frame", combined, "Combined")
    binary = np.zeros_like(sobel_x_binary)
    binary[((l_binary == 1) & (s_binary == 1) | (sobel_x_binary == 1))] = 1
    binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')
    # test(frame.image, "Original Frame", binary, "New Binary")

    # Perspective Transform of Combined Binary
    p_t = frame.perspective_transform(binary)
    p_t = frame.crop(p_t)
    # test(frame.image, "Original Frame", p_t, "Perspective Transform and Crop")

    # Find Lane Lines or return untouched and undistorted frame
    if lane.find_lines(lane_left, lane_right, p_t, image):
        return lane.draw(lane_left, lane_right, frame)
    else:
        return frame.undist

# Calibrate Camera
cam.calibrate(calibrate_dir, chessboard)

# Single Image Test Code
# out = process_image(test_img)
# plt.imshow(out)
# plt.show()

# Video Clip Pipeline
clip = VideoFileClip(input_video)
out_clip = clip.fl_image(process_image)
out_clip.write_videofile(output_video, audio=False)
