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

def diff_Image(kitty):
    '''This Differentites the Image in the X, Y Direcions'''
    X_Dir = diffXDirection(kitty)
    Y_Dir = diffYDirection(kitty)

    # Now Combine the Two Images
    diff_kitty = X_Dir + Y_Dir # Should be the same size
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
            kitty_applied[x-1][y-1] = result

    print(kitty_applied)
    return kitty_applied


# Reads the kitty => Example to check thresholding works
kitty = read_kitty("kitty.bmp")

save_kitty(smooth_constant(kitty), 'constant_blur/kitty_blurred.bmp')
save_kitty(smooth_weighted(kitty), 'weighted_blur/kitty_weighted_blur.bmp')