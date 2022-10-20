# Company: UPrintWeFix
# Author: Aiden J O'Neil

# Usage: python label.py [FOLDER NAME] [TIME OF SHIFT]
# Time of shift has format HH:MM:SS

import sys, os, shutil, csv
from pathlib import Path

CSV_FILE_NAME = "labels.csv"

# Command Line Inputs
folder = sys.argv[1]
shiftStart = sys.argv[2]

# Create writeable CSV
with open(folder + '/' + CSV_FILE_NAME, 'w', newline='\n') as csvfile:
    fieldnames = ['FILE NAME', 'GOOD OR BAD?']
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

        # Include for renaming of files
        # data_file = Path(folder + '/' + fileName)
        # good = fileName[:-4] + "_GOOD" + fileName[-4:]
        # bad = fileName[:-4] + "_BAD" + fileName[-4:]

        # If image was taken after "shiftStart", it is bad. Otherwise, it is good
        if (fileHour > shiftStart[0:2]) or (fileHour == shiftStart[0:2] and fileMin > shiftStart[3:5]) or (fileMin == shiftStart[3:5] and fileSec > shiftStart[6:8]):
            # data_file.rename(bad)
            # shutil.move(bad, folder)
            writer.writerow({"FILE NAME": fileName, "GOOD OR BAD?" : "BAD"})
        else:
            # data_file.rename(good)
            # shutil.move(good, folder)
            writer.writerow({"FILE NAME": fileName, "GOOD OR BAD?" : "GOOD"})
            