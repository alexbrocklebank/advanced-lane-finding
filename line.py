import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt


# Lane Line class
class Line:
    def __init__(self, n=5):
        # Number of line fits stored in buffer
        self.num_buffered = 0
        # Buffer max for previous line fits
        self.n = n
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = deque([], maxlen=n)
        # coefficients of the last n fits of the line
        self.recent_fitted = deque([], maxlen=n)
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # always the same y-range as image
        # self.fit_yvals = np.linspace(0, 100, num=101) * 7.2
        self.fit_yvals = np.linspace(0, 719, num=720)
        #
        self.average_x = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #
        self.current_fit_xvals = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # line position in pixels at bottom of image
        self.line_base_pos = None
        # Vehicle from center of lane
        self.vehicle_center = None
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
        # Line indices
        self.line_indices = []

    def curve_check(self, new_radius):
        if self.radius_of_curvature is None:
            return True

        self.radius_change = abs(
            new_radius - self.radius_of_curvature) / self.radius_of_curvature
        return self.radius_change <= self.radius_variance

    def remove_outliers(self):
        if (self.allx.shape == (0,)) or (self.ally.shape == (0,)):
            return self.allx, self.ally
        print("X List shape = ", self.allx.shape, ", Y List shape = ", self.ally.shape)
        mu_x, mu_y = np.mean(self.allx), np.mean(self.ally)
        sig_x, sig_y = np.std(self.allx), np.std(self.ally)
        new_x, new_y = zip(*[(x, y) for (x, y) in zip(self.allx, self.ally)
                             if abs(x - mu_x) < 2 * sig_x and abs(
                y - mu_y) < 2 * sig_y])
        return new_x, new_y

    def set_average_x(self):
        fits = self.recent_xfitted
        if len(fits) > 0:
            average = 0
            for fit in fits:
                average += np.array(fit)
            average /= len(fits)
            self.average_x = average

    def check_line(self):
        maximum_distance = 2.8
        if abs(self.vehicle_center) > maximum_distance:
            return False
        if self.num_buffered > 0:
            delta = self.diffs / self.best_fit
            if not (abs(delta) < np.array([0.7, 0.5, 0.15])).all():
                return False
        return True

    def set_averages(self):
        # Determine and Set the average of recent_xfitted
        if len(self.recent_xfitted) > 0:
            average = 0
            for fit in self.recent_xfitted:
                average += np.array(fit)
            average /= len(self.recent_xfitted)
            self.average_x = average

        # Determine and Set the average coefficients
        if len(self.recent_fitted) > 0:
            average = 0
            for coeff in self.recent_fitted:
                average += np.array(coeff)
            average /= len(self.recent_fitted)
            self.best_fit = average

    def update(self, lane):
        self.ally, self.allx = (lane[:, :, 0] > 254).nonzero()
        self.current_fit = np.polyfit(self.ally, self.allx, 2)

        yvals = self.fit_yvals
        self.current_fit_xvals = self.current_fit[0] * yvals ** 2 + \
                                 self.current_fit[1] * yvals + \
                                 self.current_fit[2]

        # Define y-value where we want radius of curvature (choose image bottom)
        y_eval = max(self.fit_yvals) * (30 / 720)
        if self.best_fit is not None:
            self.radius_of_curvature = ((1 + (2 * self.best_fit[0] * y_eval +
                                              self.best_fit[1]) ** 2) ** 1.5) \
                                       / np.absolute(2 * self.best_fit[0])

        y_eval = max(self.fit_yvals)
        self.line_base_pos = self.current_fit[0]*y_eval**2 \
                        + self.current_fit[1]*y_eval \
                        + self.current_fit[2]
        base_pos = 640

        # 3.7 meters is about 700 pixels in the x direction
        self.vehicle_center = (self.line_base_pos - base_pos) * 3.7 / 700.0

        if self.num_buffered > 0:
            self.diffs = self.current_fit - self.best_fit
        else:
            self.diffs = np.array([0, 0, 0], dtype='float')

        if self.check_line():
            self.detected = True

            self.recent_xfitted.appendleft(self.current_fit_xvals)
            self.recent_fitted.appendleft(self.current_fit)
            assert len(self.recent_xfitted) == len(self.recent_fitted)
            self.num_buffered = len(self.recent_xfitted)

            self.set_averages()

        else:
            self.detected = False

            if self.num_buffered > 0:
                self.recent_xfitted.pop()
                self.recent_fitted.pop()
                assert len(self.recent_xfitted) == len(self.recent_fitted)
                self.num_buffered = len(self.recent_xfitted)

            if self.num_buffered > 0:
                self.set_averages()

        return self.detected, self.num_buffered
