from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter

# TODO :: Figure out how to apply a kernel
# TODO :: Learn how to normalise an image

def read_kitty(path):
    # Read the Image File
    im = Image.open(path)
    # Convert this to a numpy array
    kitty = np.array(im)
    return kitty
    
def save_kitty(kitty, image_path):
    image = Image.fromarray(kitty).convert('L')

    image.save(image_path)

def smooth_constant(kitty):
    '''This Smooths the Image, Using a Constant Filter.
            Uses the Kernel:    [1, 1, 1]
                                [1, 1, 1]
                                [1, 1, 1]'''
    kernel = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
    # Smooth the kitty based on the provided kernel (Mean)
    smoothed_kitty = apply_kernel(kitty, kernel)

    return smoothed_kitty

def smooth_weighted(kitty):
    '''Uses the Smoothing Kernel:
            [1, 2, 1]
            [2, 4, 2]
            [1, 2, 1]'''
    kernel = np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]]) # This may need to be normalised after
    # Smooth the kitty for with this kernel
    smoothed_kitty = apply_kernel(kitty, kernel)

    return smoothed_kitty

def threshold_image(kitty, threshold):
    '''This sets every value above the thresold to 255, below to 0'''
    for x in range(len(kitty)):
        for y in range(len(kitty[0])):
            if kitty[x][y] > threshold:
                # Set this value to the max
                kitty[x][y] = 255
            else:
                # Set to the min
                kitty[x][y] = 0

    return kitty

def diffXDirection(kitty):
    '''This Differentiaties the Image in the X Direction (TODO::CHECK)'''
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # Now Apply the Filter
    diff_X_kitty = apply_kernel(kitty, kernel)

    return diff_X_kitty

def diffYDirection(kitty):
    '''This Differentiates the Image in the Y Direction'''
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # Now Apply the Filter
    diff_Y_kitty = apply_kernel(kitty, kernel)

    return diff_Y_kitty

def pad_edges(kitty, num):
    '''Gives the Kitty Image and the Number to Pad it with'''
    # Create an array of the number specified (probably 0)
    row1 = np.full(kitty.shape[1], int(num)).reshape(1, -1)
    row2 = row1.copy()

    # Then add the rows and columns of zeros
    kitty = np.vstack((row1, kitty))
    kitty = np.vstack((kitty, row2))

    col1 = int(num) * np.ones((kitty.shape[0], 1))
    col2 = col1.copy()

    kitty = np.hstack((col1, kitty))
    kitty = np.hstack((kitty, col2))

    return kitty

def apply_kernel(kitty, kernel):
    '''Applys the kernel provided to the .bmp image provided.'''
    # Create an array filled with zeros
    kitty_applied = np.zeros_like(kitty)

    kitty = pad_edges(kitty, 0)

    for x in range(1, kitty.shape[0] -1):
        for y in range(1, kitty.shape[1] -1):
            result = 0
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    result = result + (kernel[i][j] * kitty[x-1+i][y-1+j])
            kitty_applied[x-1][y-1] = abs(result)
    return kitty_applied

def image_magnitude(kittyx, kittyy):
    '''Computes the edge magnitude of an image'''
    kitty_mag = np.zeros_like(kittyx)

    for x in range(len(kittyx)):
        for y in range(len(kittyx[0])):
            kitty_mag[x][y] = ((kittyx[x][y] ** 2) + (kittyy[x][y] ** 2)) ** 0.5

    return kitty_mag

def process_image(smoothed_kitty, output_path):

    if output_path != 'original':
        save_kitty(smoothed_kitty, output_path + '/kitty_blurred.bmp')

    # Now Differentiate the Images, smoothed first
    smoothed_DX_kitty = diffXDirection(smoothed_kitty)
    smoothed_DY_kitty = diffYDirection(smoothed_kitty)
    # Save the Images
    save_kitty(smoothed_DX_kitty, output_path + '/kitty_DX.bmp')
    save_kitty(smoothed_DY_kitty, output_path + '/kitty_DY.bmp')
    # Get the Image Magnitude
    image_mag = image_magnitude(smoothed_DX_kitty, smoothed_DY_kitty)
    save_kitty(image_mag, output_path + '/kitty_image_mag.bmp')

    plot_histogram(image_mag, output_path)

    thresholded = threshold_image(image_mag, 210)
    save_kitty(thresholded, output_path + '/kitty_thresholded.bmp')

def plot_histogram(kitty, output_path):
    plt.cla()
    kitty_new = [element for row in kitty for element in row]

    counts = Counter(kitty_new)
    
    sorted_by_element = sorted(counts.items(), key=lambda x: x[0])

    nums = [sorted_by_element[1] for sorted_by_element in sorted_by_element]

    plt.plot(nums)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count")
    plt.savefig(output_path + '/figs.png')
    
def main():

    kitty = read_kitty("kitty.bmp")
        
    smoothed_kitty = smooth_constant(kitty)
    save_kitty(smoothed_kitty,  'constant_blur/kitty_blurred.bmp')

    wsmoothed_kitty = smooth_weighted(kitty)
    save_kitty(smoothed_kitty,  'weighted_blur/kitty_blurred.bmp')
        
    process_image(smoothed_kitty, 'constant_blur')
    process_image(wsmoothed_kitty, 'weighted_blur')
    process_image(kitty, 'original')

main()
