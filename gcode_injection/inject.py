# Title: inject.py
# Purpose: Creates and injects artificial layer-shifiting gcode at a random layer, in a random
#          direction (X or Y), with a random amount.
# Developers: Team UPrintWeFix
# Last Modified: September 22nd, 2022

# STATUS: Successfully injects layer shifting gcode into a gcode file.
# TODO: Allow shifts to subtract OR add from current position. Currently only subtracts, resulting in a
#       left shift.

import argparse
from time import time
import random

random.seed(a=None, version=2)

parser = argparse.ArgumentParser(description='Injects layer shifting GCODE into a file.')
parser.add_argument('input_filename', metavar='inp_file', type=str, help='input gcode file')

args = parser.parse_args()

input_file = args.input_filename

# Define some constants
SHIFT_AMOUNT_LOW = 2
SHIFT_AMOUNT_HIGH = 5

NUM_LAYERS = 100

LAYER_HEIGHT = 0.3 # The layer height of the print in mm

MIN_HEIGHT = LAYER_HEIGHT * 4
MAX_HEIGHT = LAYER_HEIGHT * NUM_LAYERS    # The maximum layer height of the print



# Generate random data
layer_z = round(random.uniform(MIN_HEIGHT, MAX_HEIGHT), 1)

while round(int(layer_z *10) % 2) != 0:
    layer_z = round(random.uniform(1, 10), 1)

# If layer_z is an integer, make it an integer (for gcode formatting purposes)
if int(layer_z) == float(layer_z):
    layer_z = int(layer_z)

direction = 'X' if random.random() > 0.5 else 'Y'  # Direction to shift in, either X or Y
add_sub = 'add' if random.random() > 0.5 else 'sub' # Indicator whether or not to add or subtract the shift amount

shift_amount = round(random.uniform(SHIFT_AMOUNT_LOW, SHIFT_AMOUNT_HIGH), 2)
if add_sub == 'add':
    shift_amount = -1 * shift_amount

# Open GCODE and parse
gcode = ''

with open(input_file) as f:
    gcode = f.readlines()

layer_line = None

for idx, line in enumerate(gcode):
    if 'Z:' + str(layer_z) in line:
        layer_line = idx
        break

# print(layer_line)
# print(gcode[layer_line])

X_index = gcode[layer_line-2].find(direction)
space_index = gcode[layer_line-2][X_index:].find(' ') + X_index

last_x_value = gcode[layer_line-2][X_index+1:space_index]

gcode.insert(layer_line + 1, ';----- INJECTED GCODE BEGINS HERE -----\n')
gcode.insert(layer_line + 2, ';Layer shift: layer Z=' + str(layer_z) + 'mm, direction=' + str(direction) + ', amt=' + str(shift_amount) + 'mm\n')
gcode.insert(layer_line + 3, 'G1 ' + direction + str(float(last_x_value) + shift_amount) + '\n')
gcode.insert(layer_line + 4, 'G92 ' + direction + last_x_value + '\n')
gcode.insert(layer_line + 5, ";----- INJECTED GCODE ENDS HERE-----\n")

output_file = 'Z' + str(layer_z) + '-' + str(direction) + '-' + str(shift_amount) + '.gcode'

with open(output_file, 'w') as f:
    f.writelines(gcode)


print('[INFO] Injected layer shifting gcode.')
print(' Layer: Z = ' + str(layer_z) + ' mm')
print(' Direction: ' + str(direction))
print(' Amount: ' + str(shift_amount) + ' mm')
print('\n[INFO] File saved to ' + str(output_file))


