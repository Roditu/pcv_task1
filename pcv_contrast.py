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

############
# Contrast #
############
# rumus transformasi
def T (r , r1 , s1 , r2 , s2 ) :
    s =0
    if (0 < r ) &( r < r1 ) :
        s = s1 / r1 * r
    elif ( r1 <= r ) &( r < r2 ) :
        s =( s2 - s1 ) /( r2 - r1 ) *( r - r1 ) + s1
    elif ( r2 <= r ) &( r <=255) &( r2 <255) :
        s =(255 - s2 ) /(255 - r2 ) *( r - r2 ) + s2
    else :
        s = s2
    s = np . uint8 ( np . floor ( s ) )
    return s

# Mendapatkan informasi tinggi dan lebar gambar
tinggi = image.shape[0]
lebar = image.shape[1]

# Split BGR di dari image
Bluei, Greeni, Redi = cv2.split(image)

im_B = np.zeros((tinggi,lebar), np.uint8)
im_G = np.zeros((tinggi,lebar), np.uint8)
im_R = np.zeros((tinggi,lebar), np.uint8)

# #Parameter kontras blue
r1_b = 0
s1_b = 128
r2_b = 255
s2_b = 160

for i in range ( tinggi ) :
    for j in range ( lebar ) :
        r_B = Bluei [i , j ]
        im_B [i , j ]= T (r_B , r1_b , s1_b , r2_b , s2_b )

# #Parameter kontras green
r1_g = 240
s1_g = 224
r2_g = 240
s2_g = 255

for i in range ( tinggi ) :
    for j in range ( lebar ) :
        r_G = Greeni [i , j ]
        im_G [i , j ]= T (r_G , r1_g , s1_g , r2_g , s2_g )

# #kontras red
r1_r = 100
s1_r = 20
r2_r = 175
s2_r = 200

for i in range ( tinggi ) :
    for j in range ( lebar ) :
        r_R = Redi [i , j ]
        im_R [i , j ]= T (r_R , r1_r , s1_r , r2_r , s2_r )

# Gabungkan ketiga warna gambar menjadi satu kembali
contrasted_image = cv2.merge([im_B, im_G, im_R])

# Hasil gambar dan histogram
cv2.imshow("constrast image",contrasted_image)

Cblue_hist = cv2.calcHist([contrasted_image], [0], None, [256], [0, 255])
Cgreen_hist = cv2.calcHist([contrasted_image], [1], None, [256], [0, 255])
Cred_hist = cv2.calcHist([contrasted_image], [2], None, [256], [0, 255])


# display contrasted histogram red
plt.figure("Contrasted Histogram")
plt.subplot(3, 1, 1)
plt.plot(Cred_hist, color='r')
plt.xlim([0, 255])
plt.title('red histogram')
plt.xlabel("Intensity")
plt.ylabel("Pixel Freq")

# display contrasted histogram green
plt.subplot(3, 1, 2)
plt.plot(Cgreen_hist, color='g')
plt.xlim([0, 255])
plt.title('green histogram')
plt.xlabel("Intensity")
plt.ylabel("Pixel Freq")

# display contrasted histogram blue
plt.subplot(3, 1, 3)
plt.plot(Cblue_hist, color='b')
plt.xlim([0, 255])
plt.title('blue histogram')
plt.xlabel("Intensity")
plt.ylabel("Pixel Freq")
plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()