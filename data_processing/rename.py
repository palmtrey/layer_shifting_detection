# Usage:
# python rename.py [DATA_FOLDER] [OUTPUT_FOLDER] [FILE_TYPE]
# The data folder is the directory for the images (ex. data\shift_sept30_2)
# The output folder is assumed to be in an already existing directory named automation_dataset
# Example: python rename.py data\correct_sept29_1 ender_20 jpg

# Project Directory should look like the following:
# Project
# - automation_dataset
#   - This is where your output folders will go (ex. ender_19, ender_20). These subfolders will be created for you
# - data
#   - This is where your data folders will go (ex. shift_oct1_0)
# - rename.py

import sys, os, shutil, json, random, csv, argparse
from pathlib import Path

# Command Line Inputs
parser = argparse.ArgumentParser(description='Renames data folders to standard formatting')
parser.add_argument('dataFolder', action="store", type=str, help='Input data folder')
parser.add_argument('outputFolder', action="store", type=str, help='Set output folder')
parser.add_argument('fileType', action="store", type=str, help='Set file type ex. jpg')
args = parser.parse_args()

dataFolder = args.dataFolder
outputFolder = args.outputFolder
fileType = args.fileType.lower()

# Creates directory
path = os.path.join("automation_dataset", outputFolder)
os.mkdir(path)

# Creates sorted list of images
images = os.listdir(dataFolder)
images.sort()

i = 0
for image in images:

    # Skips JSON files
    if not image.lower().endswith(fileType):
        continue

    # Renames and moves image
    new_name = outputFolder + "_" + str(i) + "." + fileType
    src = os.path.join(dataFolder, image)
    dest = os.path.join(path, new_name)

    os.rename(src, dest)
    i += 1
