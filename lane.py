import numpy as np
import cv2
import matplotlib.pyplot as plt
from line import Line

class Lane():
    def __init__(self):
        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 50
        # Choose the number of sliding windows
        self.nwindows = 9


    def find_lines(s, left, right, binary_warped, image, vis=False):
        # Identify the x and y positions of all nonzero pixels in the image
        s.nonzero = binary_warped.nonzero()
        s.nonzeroy = np.array(s.nonzero[0])
        s.nonzerox = np.array(s.nonzero[1])
        # Create empty lists to receive left and right lane pixel indices
        s.left_lane_inds = []
        s.right_lane_inds = []


        # If there is a previous frame
        if right.detected == True and left.detected == True:
            # Assume you now have a new warped binary image
            # from the next frame of video (also called "binary_warped")
            # It's now much easier to find line pixels!

            s.left_lane_inds = ((s.nonzerox > (left.best_fit[0]*(s.nonzeroy**2) + left.best_fit[1]*s.nonzeroy +
                                left.best_fit[2] - s.margin)) & (s.nonzerox < (left.best_fit[0]*(s.nonzeroy**2) +
                                left.best_fit[1]*s.nonzeroy + left.best_fit[2] + s.margin)))

            s.right_lane_inds = ((s.nonzerox > (right.best_fit[0]*(s.nonzeroy**2) + right.best_fit[1]*s.nonzeroy +
                                 right.best_fit[2] - s.margin)) & (s.nonzerox < (right.best_fit[0]*(s.nonzeroy**2) +
                                 right.best_fit[1]*s.nonzeroy + right.best_fit[2] + s.margin)))

            # Extract left and right line pixel positions
            left.allx = s.nonzerox[s.left_lane_inds]
            left.ally = s.nonzeroy[s.left_lane_inds]
            right.allx = s.nonzerox[s.right_lane_inds]
            right.ally = s.nonzeroy[s.right_lane_inds]

            minimum_indices = 10
            if left.ally.shape[0] < minimum_indices or right.ally.shape[0] < minimum_indices:
                # Detection Failed, use Window detection
                left.detected = False
                right.detected = False

        if left.detected == False or right.detected == False:
            # If either is undetected


            # Take a histogram of the bottom half of the image
            s.histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(s.histogram.shape[0]/2)
            s.leftx_base = np.argmax(s.histogram[100:midpoint]) + 100
            s.rightx_base = np.argmax(s.histogram[midpoint:-100]) + midpoint

            # Set height of windows
            s.window_height = np.int(binary_warped.shape[0]/s.nwindows)

            # Current positions to be updated for each window
            s.leftx_current = s.leftx_base
            s.rightx_current = s.rightx_base

            # Step through the windows one by one
            for window in range(s.nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*s.window_height
                win_y_high = binary_warped.shape[0] - window*s.window_height
                win_xleft_low = s.leftx_current - s.margin
                win_xleft_high = s.leftx_current + s.margin
                win_xright_low = s.rightx_current - s.margin
                win_xright_high = s.rightx_current + s.margin
                # Draw the windows on the visualization image
                if vis:
                    s.out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
                    cv2.rectangle(s.out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                                  (0,255,0), 2)
                    cv2.rectangle(s.out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                                  (0,255,0), 2)
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((s.nonzeroy >= win_y_low) & (s.nonzeroy < win_y_high) &
                (s.nonzerox >= win_xleft_low) &  (s.nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((s.nonzeroy >= win_y_low) & (s.nonzeroy < win_y_high) &
                (s.nonzerox >= win_xright_low) &  (s.nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                s.left_lane_inds.append(good_left_inds)
                s.right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > s.minpix:
                    leftx_current = np.int(np.mean(s.nonzerox[good_left_inds]))
                if len(good_right_inds) > s.minpix:
                    rightx_current = np.int(np.mean(s.nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            s.left_lane_inds = np.concatenate(s.left_lane_inds)
            s.right_lane_inds = np.concatenate(s.right_lane_inds)

            left.detected = True
            right.detected = True


        # Extract left and right line pixel positions
        left.allx = s.nonzerox[s.left_lane_inds]
        left.ally = s.nonzeroy[s.left_lane_inds]
        right.allx = s.nonzerox[s.right_lane_inds]
        right.ally = s.nonzeroy[s.right_lane_inds]

        minimum_indices = 10
        if left.ally.shape[0] < minimum_indices or right.ally.shape[0] < minimum_indices:
            # Detection Failed, use Window detection
            left.detected = False
            right.detected = False

        # Fit a second order polynomial to each
        left.best_fit = np.polyfit(left.ally, left.allx, 2)
        #print("Left Best Fit: {}".format(left.best_fit))
        right.best_fit = np.polyfit(right.ally, right.allx, 2)
        #print("Right Best Fit: {}".format(right.best_fit))

        # Generate x and y values for plotting
        s.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left.bestx = left.best_fit[0]*s.ploty**2 + left.best_fit[1]*s.ploty + left.best_fit[2]
        right.bestx = right.best_fit[0]*s.ploty**2 + right.best_fit[1]*s.ploty + right.best_fit[2]

        # Update the lines to show they have been found
        #if left.best_fit.any():
        #    left.detected = True
        #if right.best_fit.any():
        #    right.detected = True

        if vis:
            # Create an image to draw on and an image to show the selection window
            s.window_img = np.zeros_like(s.out_img)
            # Color in left and right line pixels



            s.out_img[s.nonzeroy[s.left_lane_inds], s.nonzerox[s.left_lane_inds]] = [255, 0, 0]
            s.out_img[s.nonzeroy[s.right_lane_inds], s.nonzerox[s.right_lane_inds]] = [0, 0, 255]
            plt.imshow(s.out_img)
            plt.plot(left.bestx[:], s.ploty, color='yellow')
            plt.plot(right.bestx[:], s.ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left.bestx-s.margin, s.ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left.bestx+s.margin,
                                          s.ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right.bestx-s.margin, s.ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right.bestx+s.margin,
                                          s.ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(s.window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(s.window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(s.out_img, 1, s.window_img, 0.3, 0)
            plt.imshow(result)
            plt.plot(left.bestx, s.ploty, color='yellow')
            plt.plot(right.bestx, s.ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()

        # Generate some fake data to represent lane-line pixels
        s.ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
        quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
        # For each y position generate random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        s.leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
                                      for y in s.ploty])
        s.rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
                                        for y in s.ploty])

        s.leftx = s.leftx[::-1]  # Reverse to match top-to-bottom in y
        s.rightx = s.rightx[::-1]  # Reverse to match top-to-bottom in y

        # Fit a second order polynomial to pixel positions in each fake lane line
        s.left_fit = np.polyfit(s.ploty, s.leftx, 2)
        s.left_fitx = left.best_fit[0]*s.ploty**2 + left.best_fit[1]*s.ploty + left.best_fit[2]
        s.right_fit = np.polyfit(s.ploty, s.rightx, 2)
        s.right_fitx = right.best_fit[0]*s.ploty**2 + right.best_fit[1]*s.ploty + right.best_fit[2]

        # Plot up the fake data
        if vis:
            mark_size = 3
            plt.plot(s.leftx, s.ploty, 'o', color='red', markersize=mark_size)
            plt.plot(s.rightx, s.ploty, 'o', color='blue', markersize=mark_size)
            plt.xlim(0, 1280)
            plt.ylim(0, 720)
            plt.plot(s.left_fitx, s.ploty, color='green', linewidth=3)
            plt.plot(s.right_fitx, s.ploty, color='green', linewidth=3)
            plt.gca().invert_yaxis() # to visualize as we do the images
            plt.show()

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(s.ploty)
        left.radius_of_curvature = ((1 + (2*s.left_fit[0]*y_eval + s.left_fit[1])**2)**1.5) / np.absolute(2*s.left_fit[0])
        right.radius_of_curvature = ((1 + (2*s.right_fit[0]*y_eval + s.right_fit[1])**2)**1.5) / np.absolute(2*s.right_fit[0])
        #print(left.radius_of_curvature, right.radius_of_curvature)
        # Example values: 1926.74 1908.48

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(s.ploty*ym_per_pix, s.leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(s.ploty*ym_per_pix, s.rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left.radius_of_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right.radius_of_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        #print(left.radius_of_curvature, 'm', right.radius_of_curvature, 'm')
        #print("Left Fit: ", s.left_fit, ", Right Fit: ", s.right_fit)
        # Example values: 632.1 m    626.2 m
        return True


    def vehicle_offset(s, undist):
        bottom_y = undist.shape[0] - 1
        bottom_x_left = s.left_fit[0]*(bottom_y**2) + s.left_fit[1]*bottom_y + s.left_fit[2]
        bottom_x_right = s.right_fit[0]*(bottom_y**2) + s.right_fit[1]*bottom_y + s.right_fit[2]
        offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2
        xm_per_pix = 3.7/700
        offset *= xm_per_pix

        return offset


    def draw(s, left, right, frame):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(frame.warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        #print("Left Best X: {}".format(left.bestx))
        #print("S.Plot-Y: {}".format(s.ploty))
        pts_left = np.array([np.transpose(np.vstack([left.bestx, s.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right.bestx, s.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, frame.Minv, (frame.width, frame.height))
        # Combine the result with the original image
        result = cv2.addWeighted(frame.undist, 1, newwarp, 0.3, 0)

        avg_curve = (left.radius_of_curvature + right.radius_of_curvature)/2
        label_str = 'Radius of Curve: %.1f m' % avg_curve
        result = cv2.putText(result, label_str, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)

        offset = s.vehicle_offset(frame.undist)
        label_str = 'Vehicle Offset from Center: %.1f m' % offset
        result = cv2.putText(result, label_str, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)

        #plt.imshow(result)
        #plt.show()
        return result
