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
        #
        self.average_x = None
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
        # Line indices
        self.line_indices = []
        #
        self.num_buffered = None

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
            average = average / len(fits)
            self.average_x = average

    def check_line(self):
        maximum_distance = 2.8
        if (abs(self.line_base_pos) > maximum_distance):
            return False
        if self.num_buffered > 0:
            delta = self.diffs / self.average_fit_coefficients
            if not (abs(delta) < np.array([0.7, 0.5, 0.15])).all():
                return False
        return True

    def update(self):
        def set_allxy(self,lane_candidate):
        self.ally,self.allx = (lane_candidate[:,:,0]>254).nonzero()

        def set_current_fit_coeffs(self):
        self.current_fit_coeffs = np.polyfit(self.ally, self.allx, 2)

        def set_current_fit_xvals(self):
        yvals = self.fit_yvals
        self.current_fit_xvals = self.current_fit_coeffs[0]*yvals**2 +
                self.current_fit_coeffs[1]*yvals + self.current_fit_coeffs[2]

        def set_radius_of_curvature(self):
        # Define y-value where we want radius of curvature (choose bottom of the image)
        y_eval = max(self.fit_yvals)
        if self.avg_fit_coeffs is not None:
            self.radius_of_curvature = ((1 + (2*self.avg_fit_coeffs[0]*y_eval + self.avg_fit_coeffs[1])**2)**1.5) \
                             /np.absolute(2*self.avg_fit_coeffs[0])

        def set_line_base_pos(self):
        y_eval = max(self.fit_yvals)
        self.line_pos = self.current_fit_coeffs[0]*y_eval**2 \
                        +self.current_fit_coeffs[1]*y_eval \
                        + self.current_fit_coeffs[2]
        basepos = 640

        self.line_base_pos = (self.line_pos - basepos)*3.7/600.0 # 3.7 meters is about 600 pixels in the x direction

        def get_diffs(self):
        if self.n_buffered>0:
            self.diffs = self.current_fit_coeffs - self.avg_fit_coeffs
        else:
            self.diffs = np.array([0,0,0], dtype='float')

        if check_lane():
            self.detected = True

            def add_data(self):
            self.recent_xfitted.appendleft(self.current_fit_xvals)
            self.recent_fit_coeffs.appendleft(self.current_fit_coeffs)
            assert len(self.recent_xfitted)==len(self.recent_fit_coeffs)
            self.n_buffered = len(self.recent_xfitted)

            set_average_x()

            def set_avgcoeffs(self):
            coeffs = self.recent_fit_coeffs
            if len(coeffs)>0:
                avg=0
                for coeff in coeffs:
                    avg +=np.array(coeff)
                avg = avg / len(coeffs)
                self.avg_fit_coeffs = avg

        else:
            self.detected = False

            def pop_data(self):
            if self.n_buffered>0:
                self.recent_xfitted.pop()
                self.recent_fit_coeffs.pop()
                assert len(self.recent_xfitted)==len(self.recent_fit_coeffs)
                self.n_buffered = len(self.recent_xfitted)

            if self.num_buffered > 0:
                set_average_x()

                def set_avgcoeffs(self):
                coeffs = self.recent_fit_coeffs
                if len(coeffs)>0:
                    avg=0
                    for coeff in coeffs:
                        avg +=np.array(coeff)
                    avg = avg / len(coeffs)
                    self.avg_fit_coeffs = avg

        return self.detected, self.num_buffered
