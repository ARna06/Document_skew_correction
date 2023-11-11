from Preprocessings import *
from Postprocessings import *
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

path = "Image/Path/Here"
image = plt.imread(path)
num_peaks = 6

binary_image = Image_pre_processings(image)
skew_angle = skewed_angle(binary_image,num_peaks)

fig, ax = plt.subplots(ncols=2, figsize=(20, 20))
ax[0].imshow(image, cmap='gray')
ax[1].imshow(rotate(image, skew_angle, cval=1), cmap='gray')
plt.show()

print("The skew angel is: ", skew_angle, " degrees")
