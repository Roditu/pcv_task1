import cv2
import numpy as np
from matplotlib import pyplot as plt

#import gambar
image = cv2.imread("sample.jpg",cv2.IMREAD_COLOR)
cv2.imshow("default",image)

#############
# Histogram #
############# 
# matplot membaca gambar RGB, jika BGR tidak terbaca dengan benar
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#calculate histogram
red_hist = cv2.calcHist([image_rgb], [0], None, [256], [0, 255])
green_hist = cv2.calcHist([image_rgb], [1], None, [256], [0, 255])
blue_hist = cv2.calcHist([image_rgb], [2], None, [256], [0, 255])

# display histogram red
plt.figure("Default Histogram")
plt.subplot(3, 1, 1)
plt.plot(red_hist, color='r')
plt.xlim([0, 255])
plt.title('red histogram')
plt.xlabel("Intensity")
plt.ylabel("Pixel Freq")

#display histogram green
plt.subplot(3, 1, 2)
plt.plot(green_hist, color='g')
plt.xlim([0, 255])
plt.title('green histogram')
plt.xlabel("Intensity")
plt.ylabel("Pixel Freq")

#display histogram blue
plt.subplot(3, 1, 3)
plt.plot(blue_hist, color='b')
plt.xlim([0, 255])
plt.title('blue histogram')
plt.xlabel("Intensity")
plt.ylabel("Pixel Freq")
plt.tight_layout()
plt.show()

##############
# brightness #
##############
## cara 1 dengan library open cv ##
alpha = 1 #kelipatan
beta = 80 #kecerahan (minus untuk meredupkan)
gamma = 0 #penjumlahan skalar di akhir perhitungan kecerahan

light_image = cv2.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), gamma, beta) #blending 2 gambar dengan auto contrast dan brightness

cv2.imshow("Cara 1 (terang)", light_image)
#calculate histogram
lred_hist = cv2.calcHist([light_image], [0], None, [256], [0, 255])
lgreen_hist = cv2.calcHist([light_image], [1], None, [256], [0, 255])
lblue_hist = cv2.calcHist([light_image], [2], None, [256], [0, 255])

# display histogram red
plt.figure("Brightned Histogram")
plt.subplot(3, 1, 1)
plt.plot(lred_hist, color='r')
plt.xlim([0, 255])
plt.title('red histogram')
plt.xlabel("Intensity")
plt.ylabel("Pixel Freq")

#display histogram green
plt.subplot(3, 1, 2)
plt.plot(lgreen_hist, color='g')
plt.xlim([0, 255])
plt.title('green histogram')
plt.xlabel("Intensity")
plt.ylabel("Pixel Freq")

#display histogram blue
plt.subplot(3, 1, 3)
plt.plot(lblue_hist, color='b')
plt.xlim([0, 255])
plt.title('blue histogram')
plt.xlabel("Intensity")
plt.ylabel("Pixel Freq")
plt.tight_layout()
plt.show()

## cara 2 dengan algoritma
dark_image = np.double(image) - 80
dark_image[dark_image>255] = 255
dark_image[dark_image<0] = 0
dark_image = np.uint8(np.floor(dark_image))

cv2.imshow("Cara 2 (gelap)", dark_image)
#calculate histogram
lred_hist = cv2.calcHist([dark_image], [0], None, [256], [0, 255])
lgreen_hist = cv2.calcHist([dark_image], [1], None, [256], [0, 255])
lblue_hist = cv2.calcHist([dark_image], [2], None, [256], [0, 255])

# display histogram red
plt.figure("Darkened Histogram")
plt.subplot(3, 1, 1)
plt.plot(lred_hist, color='r')
plt.xlim([0, 255])
plt.title('red histogram')
plt.xlabel("Intensity")
plt.ylabel("Pixel Freq")

#display histogram green
plt.subplot(3, 1, 2)
plt.plot(lgreen_hist, color='g')
plt.xlim([0, 255])
plt.title('green histogram')
plt.xlabel("Intensity")
plt.ylabel("Pixel Freq")

#display histogram blue
plt.subplot(3, 1, 3)
plt.plot(lblue_hist, color='b')
plt.xlim([0, 255])
plt.title('blue histogram')
plt.xlabel("Intensity")
plt.ylabel("Pixel Freq")
plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()