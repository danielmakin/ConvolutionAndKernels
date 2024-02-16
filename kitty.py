from PIL import Image
import numpy as np

def read_kitty():
    # Read the Image File
    im = Image.open('kitty.bmp')
    return im
    
def save_kitty(kitty):
    kitty = kitty.convert("RGB")
    kitty.save("ReadKitty.bmp")

# Reads the kitty as an array
kitty = read_kitty()
kitty = np.array(kitty)
print(kitty)