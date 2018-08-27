
# coding: utf-8

# In[1]:

import pytesseract
import numpy as np
import cv2 as cv2
import os
from PIL import Image
import argparse
import pyocr
from matplotlib import pyplot as plt

# In[30]:


## Image Processing Method #1 https://medium.freecodecamp.org/getting-started-with-tesseract-part-i-2a6a6b1cf75e


def apply_threshold(img, argument):
    switcher = {
        1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        4: cv2.threshold(cv2.medianBlur(img, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        5: cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        6: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        7: cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        8: cv2.threshold(cv2.GaussianBlur(img, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    }
    return switcher.get(argument, "Invalid method")


def get_string(img_path, method):
    # Read image using opencv
    img = cv2.imread(img_path)
    img_orig = cv2.imread(img_path)
    output_dir = "Output"

    # Extract the file name without the file extension
    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]

    # Create a directory for outputs
    output_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Rescale Image to 1.5 the previous size
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply dilation and erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    
    ## Depricated for apply_threshold method
    # Apply blur to smooth out the edges
    ##img = cv2.GaussianBlur(img, (3, 3), 0)
        
    # Apply threshold to get image with only b&w (binarization)
    ##img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    img = apply_threshold(img, method)
    
    # Save the filtered image in the output directory
    #save_path = os.path.join(output_path, file_name + "_filter_" + ".jpg")
    #cv2.imwrite(save_path, img)
    ##cv2.imshow("Output", img)

    # Plot Image
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    #Commenting out actual processing of image so filters can be 
    #result = pytesseract.image_to_string(img, lang="eng")
    return img


# In[33]:


def run_tesseract(image):
    result = []

    ##No longer needed as this was split to functiontest = "Images/picture1.jpg"
    
    result = pytesseract.image_to_string(image, lang="eng")
    
    #boxes = pytesseract.image_to_boxes(get_string(test), lang="eng")
    #print("Original Image Results")
    #pytesseract.image_to_string(Image.open(test))
    #

    # Split Results into an array
    result = result.split('\n')

    return result

# In[34]:

def remove_blanks(result):
    no_empties = []
    no_empties = list(filter(None, result))
    num_results = len(no_empties)    
    
    for x in range(num_results):
        print("Line", x, " = ", no_empties[x])
        ##if x != 0:
          ##  no_empties.append(result[x])
                   
    return no_empties
    
