import matplotlib.pyplot as plt
from skimage import io

image = io.imread('prusa.jpg')

# plotting the original image and the RGB channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
f.set_figwidth(15)
ax1.imshow(image)

# RGB channels
# CHANNELID : 0 for Red, 1 for Green, 2 for Blue. 
ax2.imshow(image[:, : , 0]) #Red
ax3.imshow(image[:, : , 1]) #Green
ax4.imshow(image[:, : , 2]) #Blue
f.suptitle('Different Channels of Image')

plt.savefig('out.jpg')

bin_image = image[:, :, 0] > 125

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f.set_figwidth(15)
ax1.imshow(image)
ax2.imshow(bin_image)
plt.savefig('out2.jpg')
