import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# Sobel matrix processes
class Frame:
    def __init__(self, image, undist):
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.image = image
        self.undist = undist
        self.colorspace = "RGB"
        self.HSV = cv2.cvtColor(undist, cv2.COLOR_RGB2HSV)
        self.HLS = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
        self.LUV = cv2.cvtColor(undist, cv2.COLOR_RGB2LUV)
        self.LAB = cv2.cvtColor(undist, cv2.COLOR_RGB2LAB)
        self.gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
        self.M = None
        self.Minv = None
        self.warped = None

    def abs_sobel_thresh(self, ch, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        # Apply threshold
        x, y = 0, 0
        if orient == 'x':
            x = 1
        elif orient == 'y':
            y = 1
        # ch = Single Color Channel
        # Take the derivative in x or y given orient = 'x' or 'y'
        # Take the absolute value of the derivative or gradient
        sobel = np.absolute(cv2.Sobel(ch, cv2.CV_64F, x, y))
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled = np.uint8(255*sobel/np.max(sobel))
        # Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
        mask = np.zeros_like(scaled)
        mask[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
        # Return the mask as binary output image
        return mask

    def mag_thresh(self, ch, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        # Apply threshold
        # ch = Single Color Channel
        # Take the gradient in x and y separately
        x_grad = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        y_grad = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the magnitude
        mag = np.sqrt(x_grad**2 + y_grad**2)
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scale = np.max(mag)/255
        mag = (mag/scale).astype(np.uint8)
        # Create a binary mask where mag thresholds are met
        mask = np.zeros_like(mag)
        mask[(mag >= mag_thresh[0]) & (mag <= mag_thresh[1])] = 1
        # Return the mask as binary_output image
        return mask

    def dir_threshold(self, ch, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate gradient direction
        # Apply threshold
        # ch = Single Color Channel
        # Take the gradient in x and y separately
        grad_x = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        grad_y = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the x and y gradients
        abs_x = np.absolute(grad_x)
        abs_y = np.absolute(grad_y)
        # Use np.arctan2(abs_sobely, abs_sobelx) to find the dir of the gradient
        direction = np.arctan2(abs_y, abs_x)
        # Create a binary mask where direction thresholds are met
        mask = np.zeros_like(ch)
        mask[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
        # Return the mask as binary_output image
        return mask

    def hls_thresh(self, img, thresh=(100, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary_output

    def perspective_transform(self, image):
        corners = np.float32([[190, 720], [589, 457], [698, 457], [1145, 720]])
        new_top_left = np.array([corners[0, 0], 0])
        new_top_right = np.array([corners[3, 0], 0])
        offset = [150, 0]

        src = np.float32([corners[0], corners[1], corners[2], corners[3]])
        dst = np.float32(
            [corners[0] + offset, new_top_left + offset, new_top_right - offset,
             corners[3] - offset])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        self.warped = cv2.warpPerspective(image, self.M, (self.width, self.height), flags=cv2.INTER_LINEAR)
        return self.warped

    def crop(self, image):
        vertices = np.array([[(0, 0), (self.width, 0), (self.width, 0),
                              (6 * self.width / 7, self.height),
                              (self.width / 7, self.height), (0, 0)]],
                            dtype=np.int32)
        mask = np.zeros_like(image)

        if len(image.shape) > 2:
            channels = image.shape[2]
            ignore_mask_color = (255,) * channels
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        return cv2.bitwise_and(image, mask)
