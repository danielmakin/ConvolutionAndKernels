from PIL import Image
import numpy as np

# TODO :: Figure out how to apply a kernel
# TODO :: Learn how to normalise an image

def read_kitty(path):
    # Read the Image File
    im = Image.open(path)

    # Convert this to a numpy array
    kitty = np.array(im)
    return kitty
    
def save_kitty(kitty):
    image = Image.fromarray(kitty)
    image.save("ReadKitty.bmp")

def smooth_constant(kitty):
    '''This Smooths the Image, Using a Constant Filter.
            Uses the Kernel:    [1, 1, 1]
                                [1, 1, 1]
                                [1, 1, 1]'''
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    # Smooth the kitty based on the provided kernel (Mean)
    smoothed_kitty = apply_kernel(kitty, kernel)

    return smoothed_kitty

def smooth_weighted(kitty):
    '''Uses the Smoothing Kernel:
            [1, 2, 1]
            [2, 4, 2]
            [1, 2, 1]'''
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) # This may need to be normalised after
    # Smooth the kitty for with this kernel
    smoothed_kitty = apply_kernel(kitty, kernel)

    return smoothed_kitty

def diff_Image(kitty):
    '''This Differentites the Image in the X, Y Direcions'''
    X_Dir = diffXDirection(kitty)
    Y_Dir = diffYDirection(kitty)

    # Now Combine the Two Images
    diff_kitty = X_Dir + Y_Dir # Should be the same size
    # Now Normalise the Result
    kitty = normalise_image(diff_kitty)
    return kitty

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

def normalise_image(kitty):
    '''This will place the values back between 0-255'''
    pass

def diffXDirection(kitty):
    '''This Differentiaties the Image in the X Direction (TODO::CHECK)'''
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # Now Apply the Filter
    diff_X_kitty = apply_kernel(kitty, kernel)

    return diff_X_kitty

def diffYDirection(kitty):
    '''This Differentiates the Image in the Y Direction'''
    kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    # Now Apply the Filter
    diff_Y_kitty = apply_kernel(kitty, kernel)

    return diff_Y_kitty

def apply_kernel(kitty, kernel):
    '''Applys the kernel provided to the .bmp image provided.'''
    # Think about when edge padding is needed
    pass


# Reads the kitty => Example to check thresholding works
kitty = read_kitty("kitty.bmp")

kitty = threshold_image(kitty, 100)

save_kitty(kitty)