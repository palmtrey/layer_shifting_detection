from datetime import datetime
import os

OUTPUT_PATH = './correct_clip_small_2_oct20_0'
CAP_INTERVAL_MS = 10000


if not os.path.isdir(OUTPUT_PATH):
	os.mkdir(OUTPUT_PATH)

for i in range(220):
	now = datetime.now()
	dtstr = now.strftime("%Y-%m-%d_%H_%M_%S")
	os.system('raspistill -o ' + dtstr + '.jpg -t ' + str(CAP_INTERVAL_MS) + ' --width 3280 --height 2464')
	os.system('sshpass -p uprintwefix1234 scp ' + dtstr + 'oneilaj@128.153.28.135:/home/oneilaj/images/unprocessed')
