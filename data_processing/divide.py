# Usage: python divide.py [DATA FOLDER] [PERCENT TRAIN]  [OUTPUT FOLDER] [FILE TYPE]

# PERCENT TRAIN must be an integer in range 0-100

# OUTPUT FOLDER must have subfolders train and val each with subfolders no_shift and shift



import sys, os, shutil, json, random, csv, argparse

from pathlib import Path



# Command Line Inputs

parser = argparse.ArgumentParser(description='Divides image sets into training and validation directories')
parser.add_argument('dataFolder', action="store", type=str, help='Input data folder')
parser.add_argument('percentTrain', action="store", type=int, help='Input percent of datasets going to training set')
parser.add_argument('outputFolder', action="store", type=str, help='Set output folder')
parser.add_argument('fileType', action="store", type=str, help='Set file type ex. jpg')
args = parser.parse_args()

dataFolder = args.dataFolder
percentTrain = args.percentTrain
outputFolder = args.outputFolder
fileType = args.fileType

# Create output directory
os.mkdir(outputFolder)
os.mkdir(os.path.join(outputFolder, "train"))
os.mkdir(os.path.join(outputFolder, "val"))
os.mkdir(os.path.join(outputFolder, "train", "no_shift"))
os.mkdir(os.path.join(outputFolder, "train", "shift"))
os.mkdir(os.path.join(outputFolder, "val", "no_shift"))
os.mkdir(os.path.join(outputFolder, "val", "shift"))


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

with open(os.path.join(outputFolder,'split.csv'), 'w', newline='\n') as csvfile:

    spamwriter = csv.writer(csvfile, delimiter=',')

    for folder in dataSubfolders:

        if folder in trainCorrectFolders or folder in trainShiftFolders:

            spamwriter.writerow([folder, "TRAIN"])

        elif (folder not in trainCorrectFolders and folder in correctSubfolders) or (folder not in trainShiftFolders and folder in shiftSubfolders):

            spamwriter.writerow([folder, "VAL"])



for folder in dataSubfolders:

    # Get JSON data from subfolder
    print('Folder: ' + str(folder))
    # jsonDirectory = dataFolder + '/' + folder + '/' + '_' + folder + '-labels.json'
    try:
        jsonDirectory = [x for x in os.listdir(os.path.join(dataFolder, folder)) if x.lower().endswith('.json')][0]
        jsonDirectory = os.path.join(dataFolder, folder, jsonDirectory)
        print(jsonDirectory)
    except IndexError:
        print('No json file found.')
    

    jsonFile = open(jsonDirectory)

    jsonData = json.load(jsonFile)



    jsonFileHour = jsonData[11:13]

    jsonFileMin = jsonData[14:16]

    jsonFileSec = jsonData[17:19]



    # For files going in Train subdirectory

    if folder in trainShiftFolders:

        for image in os.listdir(os.path.join(dataFolder,folder)):

            # Skip files that are not of the designated file type

            if not image.lower().endswith(fileType):

                continue

            

            imageHour = image[11:13]

            imageMin = image[14:16]

            imageSec = image[17:19]



            # If image was taken before the shift marked by the JSON file, put in train/no_shift

            if (imageHour < jsonFileHour) or (imageHour == jsonFileHour and imageMin < jsonFileMin) or (imageMin == jsonFileMin and imageSec < jsonFileSec):

                shutil.copy(os.path.join(dataFolder,folder,image), os.path.join(outputFolder,'train/no_shift'))

            # Otherwise, put in train/shift

            else:

                shutil.copy(os.path.join(dataFolder,folder,image), os.path.join(outputFolder,'train/shift'))



    # All files in trainCorrectFolders go in the train/no_shift directory

    elif folder in trainCorrectFolders:

        for image in os.listdir(os.path.join(dataFolder,folder)):

            # Skip files that are not of the designated file type

            if not image.lower().endswith(fileType):

                continue

            shutil.copy(os.path.join(dataFolder,folder,image), os.path.join(outputFolder,'train/no_shift'))



    # If folder is labeled correct but not going in train, they go in val/no_shift directory

    elif folder in correctSubfolders and folder not in trainCorrectFolders:

        for image in os.listdir(os.path.join(dataFolder,folder)):

            # Skip files that are not of the designated file type

            if not image.lower().endswith(fileType):

                continue

            shutil.copy(os.path.join(dataFolder,folder,image), os.path.join(outputFolder,'val/no_shift'))



    # If folder is labeled shift and not in train, they either go in val/no_shift or val/shift

    elif folder in shiftSubfolders and folder not in trainShiftFolders:

        for image in os.listdir(os.path.join(dataFolder,folder)):

            # Skip files that are not of the designated file type

            if not image.lower().endswith(fileType):

                continue

            imageHour = image[11:13]

            imageMin = image[14:16]

            imageSec = image[17:19]

            # If image was taken before the shift marked by the JSON file, put in val/no_shift

            if (imageHour < jsonFileHour) or (imageHour == jsonFileHour and imageMin < jsonFileMin) or (imageMin == jsonFileMin and imageSec < jsonFileSec):

                shutil.copy(os.path.join(dataFolder,folder,image), os.path.join(outputFolder,'val/no_shift'))

            # Otherwise, put in val/shift

            else:

                shutil.copy(os.path.join(dataFolder,folder,image), os.path.join(outputFolder,'val/shift'))
