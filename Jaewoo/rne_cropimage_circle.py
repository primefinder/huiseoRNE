#!/usr/local/bin/python3
import numpy as np
from PIL import Image, ImageDraw
import cv2
def CropImage(r):    
    # Open the input image as numpy array, convert to RGB
    img=Image.open("croped/result"+str(int((r-120)/2))+".png").convert("RGB")
    npImage=np.array(img)
    h,w=img.size

    # Create same size alpha layer with circle
    alpha = Image.new('L', img.size,0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice([h/2-r,w/2-r,h/2+r,w/2+r],0,360,fill=255)
    print(h,w)

    # Convert alpha Image to numpy array
    npAlpha=np.array(alpha)

    # Add alpha layer to RGB
    npImage=np.dstack((npImage,npAlpha))

    # Save with alpha
    Image.fromarray(npImage).save('croped/Croped_radius_'+str(int((r-120)/2))+'.png')

for i in range(140,300,20):
    CropImage(i)
