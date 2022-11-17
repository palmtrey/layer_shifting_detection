from datetime import datetime
from paramiko import SSHClient
from scp import SCPClient
import os

OUTPUT_PATH = './correct_clip_small_2_oct20_0'
CAP_INTERVAL_MS = 10000


if not os.path.isdir(OUTPUT_PATH):
	os.mkdir(OUTPUT_PATH)

ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect('aiden@128.153.28.135')
scp = SCPClient(ssh.get_transport())

for i in range(220):
	now = datetime.now()
	dtstr = now.strftime("%Y-%m-%d_%H_%M_%S")
	os.system('libcamera-jpeg -o ' + dtstr + '.jpg -t ' + str(CAP_INTERVAL_MS) + ' --width 3280 --height 2464 --gain 4 > /dev/null 2>&1')
	os.system('mv ' + dtstr + '.jpg ' + OUTPUT_PATH)
	print('Image captured.')
	src = path.join(OUTPUT_PATH, dtstr + '.jpg')
	des = path.join(images, dtstr + '.jpg')
	scp.put(src, des)
