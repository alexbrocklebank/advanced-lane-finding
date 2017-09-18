import cv2
import matplotlib.image as mpimg
import os
import numpy as np


class Camera:
    def __init__(self):
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        
    # Calibrate with images from camera_cal folder
    def calibrate(self, calibrate_dir, chessboard):
        # Create Object point Labels
        objp = np.zeros((chessboard[1] * chessboard[0], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
        # Create empty lists for object and image points
        objpoints = []
        imgpoints = []

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,
                    0.001)
        print("Calibrating...")

        # For each JPG in Calibrate Directory
        for f in [i for i in os.listdir(calibrate_dir) if i.endswith(".jpg")]:
            image = mpimg.imread("{}{}".format(calibrate_dir, f))
            print("Processing {}...".format(f))
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard, None)
            if ret:
                # Refine corners
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # Append Object and Image Points
                objpoints.append(objp)
                imgpoints.append(corners)

        # Calibrate Camera
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

    def undistort(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
