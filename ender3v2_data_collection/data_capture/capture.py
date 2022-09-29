from datetime import datetime
import os

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()           
drive = GoogleDrive(gauth)

upload_file_list = ['images/2022-09-20_17:03:12.jpg', 'images/2022-09-20_17:03:14.jpg']
for upload_file in upload_file_list:
	gfile = drive.CreateFile({'parents': [{'id': '17k_pTGVw_HD0j-dSb7zvyYdOGwiQGtvY'}]})
	# Read file and set it as the content of this instance.
	gfile.SetContentFile(upload_file)
	gfile.Upload() # Upload the file.

# for i in range(1000):
#     now = datetime.now()
#     dtstr = now.strftime("%Y-%m-%d_%H:%M:%S")

#     os.system('libcamera-jpeg -o ' + dtstr + '.jpg -t 500 --width 3280 --height 2464 --gain 4 > /dev/null 2>&1')
#     os.system('mv ' + dtstr + '.jpg ./images')
#     print('Image captured.')