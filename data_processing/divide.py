# Usage: python divide.py [DATA FOLDER] [PERCENT TRAIN]  [OUTPUT FOLDER]
# PERCENT TRAIN must be an integer in range 0-100
# OUTPUT FOLDER must have subfolders train and val each with subfolders no_shift and shift

import sys, os, shutil, json, random, csv
from pathlib import Path

# Command Line Inputs
dataFolder = sys.argv[1]
percentTrain = int(sys.argv[2])
outputFolder = sys.argv[3]

# Determine which folders will go to train or val
# Note: the number of folders going into train subfolder will be rounded down if percent train
#       does not evenly divide the number of data folders
#       ex. 50% of 5 data folders will put 2 folders in train
dataSubfolders = os.listdir(dataFolder)
correctSubfolders, shiftSubfolders = [], []
for folder in dataSubfolders:
    if folder[0:7] == "correct":
        correctSubfolders.append(folder)
    elif folder[0:5] == "shift":
        shiftSubfolders.append(folder)

numCorrectFolders = len(correctSubfolders)
trainCorrectFolders = random.sample(correctSubfolders, round(numCorrectFolders * percentTrain / 100))
numShiftFolders = len(shiftSubfolders)
trainShiftFolders = random.sample(shiftSubfolders, round(numShiftFolders * percentTrain / 100))

# Make CSV file
with open(outputFolder + '/split.csv', 'w', newline='\n') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for folder in dataSubfolders:
        if folder in trainCorrectFolders or folder in trainShiftFolders:
            spamwriter.writerow([folder, "TRAIN"])
        elif (folder not in trainCorrectFolders and folder in correctSubfolders) or (folder not in trainShiftFolders and folder in shiftSubfolders):
            spamwriter.writerow([folder, "VAL"])

for folder in dataSubfolders:
    # Get JSON data from subfolder
    jsonDirectory = dataFolder + '/' + folder + '/' + '_' + folder + '-labels.json'
    jsonFile = open(jsonDirectory)
    jsonData = json.load(jsonFile)

    jsonFileHour = jsonData[11:13]
    jsonFileMin = jsonData[14:16]
    jsonFileSec = jsonData[17:19]

    # For files going in Train subdirectory
    if folder in trainShiftFolders:
        for image in os.listdir(dataFolder + '/' + folder):
            # Skip JSON files
            if image[-5:] == '.json':
                continue
            
            imageHour = image[11:13]
            imageMin = image[14:16]
            imageSec = image[17:19]

            # If image was taken before the shift marked by the JSON file, put in train/no_shift
            if (imageHour < jsonFileHour) or (imageHour == jsonFileHour and imageMin < jsonFileMin) or (imageMin == jsonFileMin and imageSec < jsonFileSec):
                shutil.copy(dataFolder + '/' + folder + '/' + image, outputFolder + '/train/no_shift')
            # Otherwise, put in train/shift
            else:
                shutil.copy(dataFolder + '/' + folder + '/' + image, outputFolder + '/train/shift')

    # All files in trainCorrectFolders go in the train/no_shift directory
    elif folder in trainCorrectFolders:
        for image in os.listdir(dataFolder + '/' + folder):
            # Skip JSON files
            if image[-5:] == '.json':
                continue
            shutil.copy(dataFolder + '/' + folder + '/' + image, outputFolder + '/train/no_shift')

    # If folder is labeled correct but not going in train, they go in val/no_shift directory
    elif folder in correctSubfolders and folder not in trainCorrectFolders:
        for image in os.listdir(dataFolder + '/' + folder):
            # Skip JSON files
            if image[-5:] == '.json':
                continue
            shutil.copy(dataFolder + '/' + folder + '/' + image, outputFolder + '/val/no_shift')

    # If folder is labeled shift and not in train, they either go in val/no_shift or val/shift
    elif folder in shiftSubfolders and folder not in trainShiftFolders:
        for image in os.listdir(dataFolder + '/' + folder):
            # Skip JSON files
            if image[-5:] == '.json':
                continue
            imageHour = image[11:13]
            imageMin = image[14:16]
            imageSec = image[17:19]
            # If image was taken before the shift marked by the JSON file, put in val/no_shift
            if (imageHour < jsonFileHour) or (imageHour == jsonFileHour and imageMin < jsonFileMin) or (imageMin == jsonFileMin and imageSec < jsonFileSec):
                shutil.copy(dataFolder + '/' + folder + '/' + image, outputFolder + '/val/no_shift')
            # Otherwise, put in val/shift
            else:
                shutil.copy(dataFolder + '/' + folder + '/' + image, outputFolder + '/val/shift')
