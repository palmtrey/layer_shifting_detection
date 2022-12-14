import cv2
import json
import os
import shutil
from tqdm import tqdm

OUTPUT_DIR = 'E:/automation_dataset/images'

INPUT_DIR = 'E:/automation_dataset/mp4s/not_split/ender/camera_3/shiftdec2'
MP4_FINAL_DIR = 'E:/automation_dataset/mp4s/split'
FOLDER_NAME = 'ender'
STARTING_FOLDER_NUM = 86
META_FILE = '_meta.json'

# Metadata constants for batch
MACHINE = 'ECEender3v2'
CAMERA = 3
OBJECT = 'phil'
ERROR = 'shift'
DATE = '12-2-22'
STARTING_DATE_INSTANCE = 0


dir = os.listdir(INPUT_DIR)
dir = sorted(dir)

print(dir)

folder_num = STARTING_FOLDER_NUM - 1
date_instance = STARTING_DATE_INSTANCE - 1

print('Processing...')
for mp4 in tqdm(dir):
    folder_num += 1
    date_instance += 1

    # Get metadata

    if (ERROR == 'shift'):
        shift_height = float(mp4.split('-')[0].split('Z')[-1])
        shift_dir = mp4.split('-')[1]
        
        if len(mp4.split('-')) == 4:
            shift_amt = float(mp4.split('-')[-1].split('_')[0]) * -1
        else:
            shift_amt = float(mp4.split('-')[-1].split('_')[0]) * -1
    elif (ERROR == 'none'):
        shift_height = '-'
        shift_dir = '-'
        shift_amt = '-'

    meta_dict = {'mp4' : mp4,
                 'machine' : MACHINE,
                 'camera' : CAMERA,
                 'object' : OBJECT,
                 'error' : ERROR,
                 'date' : DATE,
                 'instance' : date_instance,
                 'shift_height' : shift_height,
                 'shift_dir' : shift_dir,
                 'shift_amt' : shift_amt}



    # exit()

    

    output_folder = FOLDER_NAME + '_' + str(folder_num)

    if not os.path.isdir(os.path.join(OUTPUT_DIR, output_folder)):
        os.mkdir(os.path.join(OUTPUT_DIR, output_folder))

    # Write metadata
    with open(os.path.join(OUTPUT_DIR, output_folder, META_FILE), 'w') as f:
        json.dump(meta_dict, f)

    capture = cv2.VideoCapture(os.path.join(INPUT_DIR, mp4))
    
    frameNr = 0

    while (True):
        success, frame = capture.read()
    
        if success:
            cv2.imwrite(os.path.join(OUTPUT_DIR, output_folder, str(output_folder) + '_' + str(frameNr) + '.jpg'), frame)
    
        else:
            break
    
        frameNr += 1
    
    capture.release()

    shutil.move(os.path.join(INPUT_DIR, mp4), os.path.join(MP4_FINAL_DIR, mp4))

    