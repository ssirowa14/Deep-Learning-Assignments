import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

#function to do convolution of 3 channel image
def convolve3d(image, temp):
	#rotate the filter 180 degree counter clockwise
    template = np.rot90(temp, 2)
    print(image.shape)
    print(template.shape)
    pad_x = int(template.shape[0]/2)
    pad_y = int(template.shape[1]/2)
    #array to store new array
    new_img = np.zeros(shape=image.shape)
    #padding
    image = np.pad(image, ((pad_x, pad_x), (pad_y, pad_y), (0,0)), 'constant')
    temp_x = template.shape[0]
    temp_y = template.shape[1]

    #for each color channel
    for k in range(3):
        for i in range(new_img.shape[0]):
            for j in range(new_img.shape[1]):
                patch = image[i:i+temp_x, j:j+temp_y, k]		#patch
                new_img[i][j][k] = np.sum(patch*template)		#convolution
            
    return new_img			#return colvoluted image


#load image
image1 = cv2.imread("Data for Hybrid Images/dog.bmp")
image2 = cv2.imread("Data for Hybrid Images/cat.bmp")

#size of guassian filter
size = 21
mask = np.zeros(shape=(size, size))
mask[int(size/2)][int(size/2)] = 1.0
mask = gaussian_filter(mask, sigma=8.0)

#low pass filtering first image
image1_new = convolve3d(image1, mask)

#high pass filtering secong image
image2_new = image2 - convolve3d(image2, mask)

#hybrid image
result = image1_new + image2_new

#normalization
for i in range(3):
#   print(np.min(result[:, :, i]), np.max(result[:, :, i]), result[:, :, i].shape)  
    image1_new[:, :, i] = (image1_new[:, :, i]-np.min(image1_new[:, :, i]))/(np.max(image1_new[:, :, i])-np.min(image1_new[:, :, i]))
    image2_new[:, :, i] = (image2_new[:, :, i]-np.min(image2_new[:, :, i]))/(np.max(image2_new[:, :, i])-np.min(image2_new[:, :, i]))
    result[:, :, i] = (result[:, :, i]-np.min(result[:, :, i]))/(np.max(result[:, :, i])-np.min(result[:, :, i]))

# plt.imshow(image1_new)
# plt.show()

# plt.imshow(image2_new)
# plt.show()

#show the image
plt.imshow(result)
plt.show()