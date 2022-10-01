# Import packages
import cv2
import numpy as np
import os
from tqdm import tqdm

<<<<<<< Updated upstream:data_processing/crop.py
DATA_PATH = './spaghetti_cap_sept28_10s'
OUTPUT_PATH = './spaghetti_cap_sept28_10s_cropped'
=======
DATA_PATH = '/home/cameron/Documents/image_data/original/shift_cap_sept29_10s'
OUTPUT_PATH = '/home/cameron/Documents/image_data/cropped/shift_cap_sept29_10s_cropped'
SCALED_IMAGE_SIZE = (224, 224)

>>>>>>> Stashed changes:data_processing/cropandscale.py
top_bound = int(2464/2 - 150)
bottom_bound = int(2464/2 + 200)
left_bound = int(3280/2 - 200)
right_bound = int(3280/2 + 150)


images = []

if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


for img_name in os.listdir(DATA_PATH):
    images.append(DATA_PATH + '/' + img_name)

# print(images)

for img_path in tqdm(images):

    img_name = img_path.split('/')[-1]

    img = cv2.imread(img_path)
    # print(img.shape) # Print image shape
    # cv2.imshow("original", img)

    # Cropping an image
    cropped_image = img[top_bound:bottom_bound, left_bound:right_bound]
    # print(cropped_image.shape)

    dim = (264, 264)
    cropped_image = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)

    # Display cropped image
    # cv2.imshow("cropped", cropped_image)

    # Save the cropped image
    cv2.imwrite(OUTPUT_PATH + '/' + img_name, cropped_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # break