# I don't know if I am just lazy or if this is saving me time. Either way my motivation for making this is laziness

# Usage: python create_meta.py [OUTPUT FOLDER]
# Example: python create_meta.py ender_0

# To use this program, your directory should look like the following:
# Project Folder
# - automation_dataset
#   - ender_0
#   - ender_1
#   - ender_2
#   - ...
# - 3D Printer Automation Data - Sheet1.csv
# - create_meta.py

import sys, os, shutil, json, random, csv, argparse, json
from pathlib import Path

parser = argparse.ArgumentParser(description='Creates meta files')
parser.add_argument('outputFolder', action="store", type=str, help='Set output folder')
args = parser.parse_args()
outputFolder = args.outputFolder

with open('3D Printer Automation Data - Sheet1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if row[0] == outputFolder:
            dictionary = {
                "mp4": row[5],
                "machine": row[6],
                "camera": row[7],
                "object": row[8],
                "error": row[9],
                "date": row[10],
                "instance": row[11],
                "shift_height": row[12],
                "shift_dir": row[13],
                "shift_amt": row[14]
            }

            json_object = json.dumps(dictionary, indent=4)
            print_id = row[0]
            json_file = os.path.join("automation_dataset", print_id, "_meta.json")

            with open(json_file, "w") as outfile:
                outfile.write(json_object)


