# Name: replace_colon.py
# Purpose: Renames image files in a given folder from
#          "%Y-%m-%d_%H:%M:%S" to "%Y-%m-%d_%H_%M_%S"
# Author: Cameron Palmer, campalme@clarkson.edu
# Last Modified: October 20th, 2022

import os

DATA_PATH = './shift_oct6_0'

for file in os.listdir(DATA_PATH):
    newname = file.replace(':','-')
    os.system('mv ' + DATA_PATH + '/' + str(file) + ' ' + DATA_PATH + '/' + str(newname))