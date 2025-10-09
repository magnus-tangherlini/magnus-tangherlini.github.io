import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage as sk
import skimage.io as skio
from skimage import io, img_as_ubyte
from skimage.transform import rescale, resize
from scipy import signal


def get_points(im1, im2):
    print("Click corresponding points between the two images (left ↔ right).")
    print("Click on Image 1, then on Image 2, alternating.")
    print("Press Enter (or right-click) when done.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(im1)
    ax1.set_title("Image 1 (Left)")
    ax2.imshow(im2)
    ax2.set_title("Image 2 (Right)")

    plt.tight_layout()

    points_1 = []
    points_2 = []

    while True:
        print("\nSelect next point on Image 1 (press Enter when done)...")
        pts1 = plt.ginput(1, timeout=0)
        if len(pts1) == 0:
            print("Finished selecting points.")
            break

        # Round to integer pixel coordinates
        x1, y1 = map(int, np.round(pts1[0]))
        points_1.append((x1, y1))
        ax1.plot(x1, y1, 'ro')
        fig.canvas.draw()

        print("Select corresponding point on Image 2...")
        pts2 = plt.ginput(1, timeout=0)
        if len(pts2) == 0:
            print("Finished selecting points.")
            break

        x2, y2 = map(int, np.round(pts2[0]))
        points_2.append((x2, y2))
        ax2.plot(x2, y2, 'ro')
        fig.canvas.draw()

    plt.close(fig)
    plt.pause(0.1)

    if len(points_1) != len(points_2):
        print("Unequal number of points — trimming to shortest list.")
        n = min(len(points_1), len(points_2))
        points_1 = points_1[:n]
        points_2 = points_2[:n]

    print(f"\nCollected {len(points_1)} point pairs.")
    print("Points Image 1:", points_1)
    print("Points Image 2:", points_2)

    return [points_1, points_2]

def get_points_for_rectification(im1):
    print("Please select 4 points in the image for rectification.")
    print("Click in the order: top-left, top-right, bottom-left, bottom-right.")
    print("Press Enter (or right-click) when done.")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(im1)
    ax.set_title("Select points for rectification")
    plt.tight_layout()

    points = []

    while len(points) < 4:
        print(f"\nSelect point {len(points)+1}...")
        pts = plt.ginput(1, timeout=0)
        if len(pts) == 0:
            print("Finished selecting points early.")
            break

        # Round to integer pixel coordinates
        x, y = map(int, np.round(pts[0]))
        points.append((x, y))

        # Plot the point
        ax.plot(x, y, 'ro')
        fig.canvas.draw()

    plt.close(fig)
    plt.pause(0.1)

    print(f"\nCollected {len(points)} points for rectification.")
    print("Points:", points)

    return points

def gaussian_stack_rgb(img, levels, sigma):
    stack = [img.copy()]
    #gaus_kernel = cv2.getGaussianKernel(6*sigma+1, sigma)
    #gaus_kernel_2d = gaus_kernel @ gaus_kernel.T
    #gaus_kernel_2d /= np.sum(gaus_kernel_2d)
    for i in range(0, levels):
        get_prev = stack[i]
        sigma_i = sigma #* (2 ** i)
        gaus_kernel = cv2.getGaussianKernel(6*sigma_i+1, sigma_i)
        gaus_kernel_2d = gaus_kernel @ gaus_kernel.T
        convolved_R = signal.convolve2d(get_prev[:,:,0], gaus_kernel_2d, mode='same', boundary='symmetric')
        #normalized_R = convolved_R / np.max(convolved_R)

        convolved_G = signal.convolve2d(get_prev[:,:,1], gaus_kernel_2d, mode='same', boundary='symmetric') 
        #normalized_G = convolved_G / np.max(convolved_G)

        convolved_B = signal.convolve2d(get_prev[:,:,2], gaus_kernel_2d, mode='same', boundary='symmetric') 
        #normalized_B = convolved_B / np.max(convolved_B)

        full_image = np.stack([convolved_R, convolved_G, convolved_B], axis = 2)
        stack.append(full_image)

    return stack
    

def laplacian_stack_rgb(img, levels, sigma):
    gaus_stack = gaussian_stack_rgb(img, levels, sigma)
    stack = []
    for i in range(1, len(gaus_stack)):
        high_freq = gaus_stack[i-1] - gaus_stack[i]
        #scaled = (high_freq - np.min(high_freq))/ (np.max(high_freq) - np.min(high_freq)) #scale back to 0 and 1
        scaled = high_freq
        stack.append(scaled)
    stack.append(gaus_stack[-1])
    return stack
    #create it using gaussian stack

def combined_color(im1, im2, mask, levels, sigma1, sigma2, sigma3):
    #create gaussians
    im1_lap = laplacian_stack_rgb(im1, levels, sigma1)
    im2_lap = laplacian_stack_rgb(im2, levels, sigma2)
    #create blurred mask
    mask_blur = gaussian_stack_rgb(mask, levels, sigma3)
    #convolve 
    saved_convolutions = []

    for i in range(len(im1_lap)):
        #compute our convolution for the ith layer, across 3 channels

        #compute our convolution for the ith layer, across 3 channels
        new_image = im1_lap[i]* mask_blur[i] + im2_lap[i] *(1 - mask_blur[i])
        #append to list

        saved_convolutions.append(new_image)
        
        #append to list

    return saved_convolutions






