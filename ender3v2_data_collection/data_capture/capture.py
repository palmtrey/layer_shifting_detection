from datetime import datetime
import os, scp


OUTPUT_PATH = './correct_clip_small_2_oct20_0'
CAP_INTERVAL_MS = 10000


if not os.path.isdir(OUTPUT_PATH):
	os.mkdir(OUTPUT_PATH)


for i in range(220):
    now = datetime.now()
    dtstr = now.strftime("%Y-%m-%d_%H_%M_%S")

    os.system('libcamera-jpeg -o ' + dtstr + '.jpg -t ' + str(CAP_INTERVAL_MS) + ' --width 3280 --height 2464 --gain 4 > /dev/null 2>&1')
    os.system('mv ' + dtstr + '.jpg ' + OUTPUT_PATH)
    print('Image captured.')

    client = scp.Client(host='128.153.28.135', user='user', password='password')
    client.transfer('/etc/local/' + OUTPUT_PATH + '/' + dtstr, '/etc/remote/'  + OUTPUT_PATH + '/' + dtstr)
