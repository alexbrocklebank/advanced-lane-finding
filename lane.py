import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from line import Line


class Lane:
    def __init__(self):
        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 15
        # Choose the number of sliding windows
        self.nwindows = 9
        # Store problematic frames
        self.issues = 0

    def compare(self, left, right):
        pass

    def find_peaks(self, image, threshold):
        half = image[image.shape[0]/2:,:,0]
        data = np.sum(half, axis=0)
        filtered = scipy.ndimage.filters.gaussian_filter1d(data, 20)
        peak_ind = signal.find_peaks_cwt(filtered, np.arange(20, 300))
        peaks = np.array(peak_ind)
        peaks = peaks[filtered[peak_ind] > threshold]
        return peaks, filtered

    def increment_window(self, image, center_point, width):
        ny, nx, _ = image.shape
        mask = np.zeros_like(image)

        if center_point <= width / 2:
            center_point = width / 2
        if center_point >= nx - width / 2:
            center_point = nx - width / 2

        left = center_point - width / 2
        right = center_point + width / 2

        vertices = np.array([[(left, 0), (left, ny), (right, ny), (right, 0)]],
                            dtype=np.int32)
        ignore_mask_color = (255, 255, 255)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked = cv2.bitwise_and(mask, img)
        histogram = np.sum(masked[:, :, 0], axis=0)
        if max(histogram > 10000):
            center = np.argmax(hist)
        else:
            center = center_point
        return masked, center

    def get_lane_binary(self, image, line, window_center, width=300):
        if line.detected:
            window_center = line.bestx
        else:
            peaks, filtered = find_peaks(image, threshold=3000)
            if len(peaks) != 2:
                print(str(len(peaks)), " lanes detected!")
                plt.imsave('problem_frame_{}.jpg'.format(self.issues), image)
                self.issues += 1
            peak_indices = np.argmin(abs(peaks - window_center))
            window_center = peaks[peak_indices]
        num_zones = 6
        ny, nx, nc = image.shape
        zones = image.reshape(num_zones, -1, nx, nc)
        zones = zones[::-1]  # started from the bottom
        window, center = increment_window(zones[0], window_center, width)
        for zone in zones[1:]:
            next_window, center = increment_window(zone, center, width)
            window = np.vstack((next_window, window))
        return window

    def find_lines(s, left, right, binary_warped, image, vis=False):

        left_window_center = 340
        left_binary = get_lane_binary(binary_warped, left, left_window_center,
                                      width=300)

        right_window_center = 940
        right_binary = get_lane_binary(binary_warped, right,
                                       right_window_center, width=300)

        left.detected, left.num_buffered = left.update(left_binary)
        right.detected, right.num_buffered = right.update(right_binary)

        return True

    def draw(s, left, right, frame):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(frame.warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left.bestx,
                                                     left.fit_yvals]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right.bestx,
                                                                right.fit_yvals]
                                                               )))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp blank back to orig img space using inv perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, frame.Minv, (frame.width,
                                                               frame.height))
        # Combine the result with the original image
        result = cv2.addWeighted(frame.undist, 1, newwarp, 0.3, 0)

        # Draw Vehicle Offset on frame
        lane_width = 3.7
        offset = -1 * round(0.5 * (right.vehicle_center - lane_width / 2) +
                              0.5 * (abs(left.vehicle_center) - lane_width /
                                     2), 2)
        label_str = 'Vehicle Offset from Center: %.1f m' % offset
        result = cv2.putText(result, label_str, (30, 70), 0, 1, (0, 0, 0), 2,
                             cv2.LINE_AA)

        # Draw Radius of Curve on frame if calculable
        if left.radius_of_curvature and right.radius_of_curvature:
            curvature = 0.5 * (round(right.radius_of_curvature, 1) +
                               round(left.radius_of_curvature, 1))
            label_str = 'Radius of Curve: %.1f m' % avg_curve
            result = cv2.putText(result, label_str, (30, 40), 0, 1, (0, 0, 0), 2,
                             cv2.LINE_AA)

        #plt.imshow(result)
        #plt.show()
        return result
