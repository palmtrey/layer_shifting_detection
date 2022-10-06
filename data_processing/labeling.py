
# Usage: python labeling.py [FOLDER NAME] [TIME OF SHIFT] [Valid OR Training]
# Time of shift has format HH:MM:SS

import sys, os, shutil, csv
from pathlib import Path

# Command Line Inputs
folder = sys.argv[1]
shiftStart = sys.argv[2]
valid_train = sys.argv[3]
CSV_FILE_NAME = "_" + folder + "-labels.csv"

# Create writeable CSV
with open(folder + '/' + CSV_FILE_NAME, 'w', newline='\n') as csvfile:
    fieldnames = ['FILE NAME', 'SHIFT OR NO SHIFT?']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Goes through every file except the CSV
    for fileName in os.listdir(folder):
        if fileName == CSV_FILE_NAME:
            continue

        # File Name has format YYYY-MM-DD_HH:MM:SS.jpg
        fileHour = fileName[-12:-10]
        fileMin = fileName[-9:-7]
        fileSec = fileName[-6:-4]

        # If image was taken after "shiftStart", it is bad. Otherwise, it is good
        if (fileHour > shiftStart[0:2]) or (fileHour == shiftStart[0:2] and fileMin > shiftStart[3:5]) or (fileMin == shiftStart[3:5] and fileSec > shiftStart[6:8]):
            # writer.writerow({"FILE NAME": fileName, "SHIFT OR NO SHIFT?" : "SHIFT"})
            shutil.move(folder + '/' + fileName, valid_train + '/shift')
        else:
            # writer.writerow({"FILE NAME": fileName, "SHIFT OR NO SHIFT?" : "NO SHIFT"})
            shutil.move(folder + '/' + fileName, valid_train + '/no_shift')