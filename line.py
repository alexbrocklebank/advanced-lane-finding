import numpy as np
import cv2
import matplotlib.pyplot as plt


# Lane Line class
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # Allowed change in radius from frame to frame (100%/X%)
        self.radius_variance = 0.5  # 200%
        # Change in radius between frames
        self.radius_change = 0

    def curve_check(self, new_radius):
        if self.radius_of_curvature is None:
            return True

        self.radius_change = abs(
            new_radius - self.radius_of_curvature) / self.radius_of_curvature
        return self.radius_change <= self.radius_variance

    @staticmethod
    def remove_outliers(x_list, y_list):
        if not x_list or not y_list:
            return x_list, y_list
        mu_x, mu_y = np.mean(x_list), np.mean(y_list)
        sig_x, sig_y = np.std(x_list), np.std(y_list)
        new_x, new_y = zip(*[(x, y) for (x, y) in zip(x_list, y_list)
                             if abs(x - mu_x) < 2 * sig_x and abs(
                y - mu_y) < 2 * sig_y])
        return new_x, new_y
