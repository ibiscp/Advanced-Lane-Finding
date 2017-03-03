import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

# Read in and make a list of calibration images
path = 'camera_cal/'
images = glob.glob(path + 'calibration*.jpg')

# Array to store object points and image points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Prepare object points
objp = np.zeros((9*6, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Configure plot
fig = plt.figure(facecolor="white")
fig.set_size_inches(15, 8)

for i in range(len(images)):
    # Read in each image
    img = cv2.imread(images[i])

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If corners are found, add object points, image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

    # Draw and display the corners
    ax=fig.add_subplot(4,5,i+1, aspect='equal');
    try:
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
    except:
        pass
    ax.imshow(img);

    if ret != True:
        ax.axhline(linewidth=5, color="r")
        ax.axvline(linewidth=5, color="r")

    ax.axis('off');
    ax.axis('tight');

plt.suptitle('Distorted chessboard with corners identified', fontsize = 20);

# Correction for distortion
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# Function to undistort a given
def undistort_image(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

# Configure plot
fig = plt.figure(facecolor="white")
fig.set_size_inches(15, 8)

for i in range(len(images)):
    img = cv2.imread(images[i])

    undist = undistort_image(img)

    ax=fig.add_subplot(4,5,i+1, aspect='equal');
    ax.imshow(undist);
    ax.axis('off');
    ax.axis('tight');

plt.suptitle('Undistorted chessboard', fontsize = 20);

import os

path = "test_images//"
path = "challenge_examples//"
image_names = os.listdir(path)

# Configure plot
fig = plt.figure(facecolor="white")
fig.set_size_inches(15, 4)

for i in range(min(len(image_names), 8)):
    img = cv2.imread(path + image_names[i])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    undistorted = undistort_image(img)

    ax=fig.add_subplot(2,4,i+1, aspect='equal');
    ax.imshow(undistorted);
    ax.axis('off');
    ax.axis('tight');

plt.suptitle('Undistorted images', fontsize = 20);

def plot_images(img1, img2, title1, title2, colormap1, colormap2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.tight_layout()
    ax1.imshow(img1, cmap=colormap1)
    ax1.set_title(title1, fontsize=30)
    ax2.imshow(img2, cmap=colormap2)
    ax2.set_title(title2, fontsize=30)
    ax1.axis('off');
    ax2.axis('off');

from IPython.html import widgets
from IPython.html.widgets import interact
from IPython.display import display

# Parameters
p1x = 545
p2x = 742
p3x = 1602
p4x = -261
pay = 450
pby = 720

# Perspective transfor function
def warp(img, p1x, p2x, p3x, p4x, pay, pby):

    img_size = (img.shape[1], img.shape[0])

    # Source coordinates
    src = np.float32([[p1x, pay], [p2x, pay], [p3x, pby], [p4x, pby]])

    # Desired coordinates
    dst = np.float32([[0, 0], [img_size[0], 0], [img_size[0], img_size[1]], [0, img_size[1]]])

    # Perspective transform
    M = cv2.getPerspectiveTransform(src, dst)

    # Inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, Minv

# Test and choose parameters
def print_image(p1x, p2x, p3x, p4x, pay, pby):
    img = cv2.imread(path + image_names[1])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # Undistort image
    img = undistort_image(img)

    # Transform
    img_warp, Minv = warp(img, p1x, p2x, p3x, p4x, pay, pby)

    # Plot
    plot_images(img, img_warp, 'Original', "Bird's-eye view", 'jet', 'jet')

# Parameters
gradx_min = 20# 40
gradx_max = 160
grady_min = 96
grady_max = 255

# Absolute sobel threshold
def sobel_threshold(img, orient, thresh_min, thresh_max):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output

# Test and choose parameters
def plot_sobel_threshold(gradx_min, gradx_max, grady_min, grady_max):
    img = cv2.imread(path + image_names[2])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # Undistort image
    img = undistort_image(img)

    # Transform
    gradx = sobel_threshold(img, orient='x', thresh_min=gradx_min, thresh_max=gradx_max)
    grady = sobel_threshold(img, orient='y', thresh_min=grady_min, thresh_max=grady_max)

    # Combine
    combined = gradx | grady

    # Plot
    plot_images(img, combined, 'Original', "Sobel threshold", 'jet', 'gray')

# Parameters
s_min = 100#120
s_max = 255
l_min = 40
l_max = 255

def color_threshold(img, s_thresh=(s_min, s_max), l_thresh=(l_min, l_max)):

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Threshold saturation channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Threshold lightness
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    binary = np.zeros_like(s_channel)
    binary[((l_binary == 1) & (s_binary == 1))] = 1
    binary = np.array(binary).astype('uint8')

    return  binary

def plot_color_threshold(s_min, s_max, l_min, l_max):
    #for image in image_names:
        img = cv2.imread(path + image_names[2])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # Undistort image
        img = undistort_image(img)

        # Color threashold
        color = color_threshold(img, s_thresh=(s_min, s_max), l_thresh=(l_min, l_max))

        # Plot
        plot_images(img, color, 'Original', "Color threashold", 'jet', 'gray')

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read in an image
img = cv2.imread(path + image_names[2])

# Convert to RGB
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# Undistort image
undistorted = undistort_image(img)

# Color threashold
color = color_threshold(undistorted)

# Sobel threashold
gradx = sobel_threshold(undistorted, orient='x', thresh_min=gradx_min, thresh_max=gradx_max)
grady = sobel_threshold(undistorted, orient='y', thresh_min=grady_min, thresh_max=grady_max)

# Combine
combined = color | gradx | grady

# Warp image
img_warp, Minv = warp(combined, p1x, p2x, p3x, p4x, pay, pby)

# Shape
shape = img_warp.shape

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read in an image
img = cv2.imread(path + image_names[2])

# Convert to RGB
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# Undistort image
undistorted = undistort_image(img)

# Color threashold
color = color_threshold(undistorted)

# Sobel threashold
gradx = sobel_threshold(undistorted, orient='x', thresh_min=gradx_min, thresh_max=gradx_max)
grady = sobel_threshold(undistorted, orient='y', thresh_min=grady_min, thresh_max=grady_max)

# Combine
combined = color | gradx | grady

# Warp image
img_warp, Minv = warp(combined, p1x, p2x, p3x, p4x, pay, pby)

# Shape
shape = img_warp.shape

def get_polifyt(binary_warped, last_fit, restart = True):

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Set the width of the windows +/- margin
    margin = 100
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    if restart:
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Choose the number of sliding windows
        nwindows = 10
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    else:
        left_fit = last_fit[0]
        right_fit = last_fit[1]

        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Draw polyline on image
    right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
    left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
    cv2.polylines(out_img, [right], False, (255,255,0), thickness=10)
    cv2.polylines(out_img, [left], False, (255,255,0), thickness=10)

    return [left_fit, right_fit], out_img

fit, out_img = get_polifyt(img_warp, [])

plot_images(img_warp, out_img, 'Warped image', "With polylines", 'gray', 'jet')

def curvature(shape, fit):
    left_fit = fit[0]
    right_fit = fit[1]

    # I'll choose the maximum y-value, corresponding to the bottom of the image
    ploty = np.linspace(0, shape[0]-1, shape[0])
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Calculate offset
    offset = (((left_fit[0]*720**2+left_fit[1]*720+left_fit[2]) +
               (right_fit[0]*720**2+right_fit[1]*720+right_fit[2]))/2 - 1280/2)*xm_per_pix

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    # Build outputs
    curvature = round((left_curverad + right_curverad)/2, 2)
    center_offset = round(offset, 2)

    return [left_curverad, right_curverad, curvature], center_offset

curvature(shape, fit)

def draw_lines(undistorted, Minv, fit):
    # Get left and right polynomial equations
    left_fit = fit[0]
    right_fit = fit[1]

    # Draw the polynomial
    ploty = np.linspace(0, undistorted.shape[0]-1, undistorted.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(undistorted[:,:,1]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

    return result

result = draw_lines(undistorted, Minv, fit)

plot_images(undistorted, result, 'Original', "With path", 'jet', 'jet')

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # # was the line detected in the last iteration?
        # self.detected = False
        # # x values of the last n fits of the line
        # self.recent_xfitted = []
        # # average x values of the fitted line over the last n iterations
        # self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        # #difference in fit coefficients between last and new fits
        # self.diffs = np.array([0,0,0], dtype='float')
        # #x values for detected line pixels
        # self.allx = None
        # #y values for detected line pixels
        # self.ally = None

def save_image(data, fn):

    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data)
    plt.savefig(fn, dpi = height)
    plt.close()

diff = list()
def sanity_check(history, line):
    global diff
    sanity = False
    if history:
        # # Maximum difference
        # diff = 50
        # # Check left polyfit
        # if abs(max(line.current_fit[0]) - max(history[-1].current_fit[0])) > diff:
        #     line.current_fit[0] = history[-1].current_fit[0]
        #     sanity += 1
        # # Check right polyfit
        # if abs(max(line.current_fit[1]) - max(history[-1].current_fit[1])) > diff:
        #     line.current_fit[1] = history[-1].current_fit[1]
        #     sanity += 1
        # if (line.radius_of_curvature - history[-1].radius_of_curvature) > 1000:
        #     sanity += 1
        if (line.current_fit[0] == line.current_fit[1]).all():
            sanity = True
        if abs(line.radius_of_curvature[0] - line.radius_of_curvature[1]) > 10000:
            sanity = True
        # Check if polynomials intersect each other
        equation = np.roots(np.subtract(line.current_fit[0], line.current_fit[1]))
        real_value = equation.real[abs(equation.imag)<1e-5]
        if len(real_value) > 0 and all(abs(i) <=1000  for i in real_value):
            sanity = True
        # Check distance on top and botton between the two lanes
        top = line.current_fit[0][2] - line.current_fit[1][2]
        botton = (line.current_fit[0][0]*720**2 + line.current_fit[0][1]*720 + line.current_fit[0][3]) - \
                 (line.current_fit[1][0]*720**2 + line.current_fit[1][1]*720 + line.current_fit[1][3])
        diff.append(abs(top - botton))

    return sanity
def build_frame(bird_img, threshold_img, polynomial_img, curv, offset, lines_img):
    # Define output image
    # Main image
    img_out=np.zeros((720,1707,3), dtype=np.uint8)
    img_out[0:720,0:1280,:] = lines_img

    # Text formatting
    fontScale=1
    thickness=2
    fontFace = cv2.FONT_HERSHEY_SIMPLEX

    # Bird's-eye view image
    img_out[0:240,1281:1707,:] = cv2.resize(bird_img,(426,240))
    boxsize, _ = cv2.getTextSize("Bird's-eye view", fontFace, fontScale, thickness)
    cv2.putText(img_out, "Bird's-eye view", (int(1494-boxsize[0]/2),40), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)

    # Threshold image
    resssized = cv2.resize(threshold_img,(426,240))
    gray_image = cv2.cvtColor(resssized*255,cv2.COLOR_GRAY2RGB)
    img_out[241:481,1281:1707,:] = cv2.resize(gray_image,(426,240))
    boxsize, _ = cv2.getTextSize("Threshold image", fontFace, fontScale, thickness)
    cv2.putText(img_out, "Threshold image", (int(1494-boxsize[0]/2),281), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)

    # Polynomial lines
    img_out[480:720,1281:1707,:] = cv2.resize(polynomial_img,(426,240))
    boxsize, _ = cv2.getTextSize("Polynomial lines", fontFace, fontScale, thickness)
    cv2.putText(img_out, "Polynomial lines", (int(1494-boxsize[0]/2),521), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)

    # Curve and center offset text
    curvature = "Curvature: " + str(curv) + ' m'
    center_offset = "Center offset: " + str(offset)  + ' m'
    cv2.putText(img_out, curvature, (40,40), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
    cv2.putText(img_out, center_offset, (40,80), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)

    # Convert to BGR
    result = cv2.cvtColor(img_out,cv2.COLOR_RGB2BGR)

    return result

import math

import math

def process_image(img):
    global history, restart

    # Create a new instance of line
    line = Line()

    if (len(history)%100 == 0):
        save_image(img, 'challenge_examples\\' + 'challenge_' + str(len(history)))

    # Convert to RGB
    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # Shape
    shape = rgb_img.shape

    # Undistort image
    undistorted_img = undistort_image(rgb_img)

    # Color threashold
    color_threshold_img = color_threshold(undistorted_img)

    # Sobel threashold
    gradx_threshold_img = sobel_threshold(undistorted_img, orient='x', thresh_min=gradx_min, thresh_max=gradx_max)
    grady_threshold_img = sobel_threshold(undistorted_img, orient='y', thresh_min=grady_min, thresh_max=grady_max)

    # Combine
    threshold_img = color_threshold_img | gradx_threshold_img | grady_threshold_img

    # Warp image
    warped_img, Minv = warp(threshold_img, p1x, p2x, p3x, p4x, pay, pby)

    # Get polynomial
    try:
        last = history[-1].current_fit
    except:
        last = None
    line.current_fit, polynomial_img = get_polifyt(warped_img, last, restart)

    # Get curvature
    line.radius_of_curvature, line.line_base_pos = curvature(shape, line.current_fit)

    ######## Call sanity check to validate last data
    if (sanity_check(history, line)):
        restart = True
        line.current_fit, polynomial_img = get_polifyt(warped_img, last, restart)
        line.radius_of_curvature, line.line_base_pos = curvature(shape, line.current_fit)
    restart = False
    history.append(line)

    # Bird's-eye view image
    bird_img, _ = warp(undistorted_img, p1x, p2x, p3x, p4x, pay, pby)



    # Draw lines on image
    lines_img = draw_lines(undistorted_img, Minv, line.current_fit)

    # Build frame
    last_integer = max(math.floor((len(history)) / 10) * 10 -1 ,0)
    curv = history[last_integer].radius_of_curvature[-1]
    center_offset = history[last_integer].line_base_pos
    result = build_frame(bird_img, threshold_img, polynomial_img, curv, center_offset, lines_img)

    return result

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

history = []
restart = True
sanity = 0
project_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(project_output, audio=False)