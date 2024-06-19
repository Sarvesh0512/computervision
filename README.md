# computervision
!pip install opencv-python
!pip install matplotlib
!pip install numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np
def load_image(path):
    image = cv2.imread(path)
    if image.shape[2]==3:
       return image
    else:
       return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


       #LOAD THE IMAGE (MODIFY PATH AS NEEDED)
image=load_image("sarvesh photo.jpg")
#grey scale conversion
gray_cv2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
weights = [0.2989,0.5870,0.1140]
gray_numpy = np.dot(image,weights).astype(np.int8)

#plot image using matplotlib
plt.figure(figsize=(10,10))

#ployt orginal image
plt.subplot(131)
plt.title("Orginal Image (BGR)")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

#plot grey scale images
plt.subplot(132)
plt.title("gray scale (open cv)")
plt.imshow(gray_cv2, cmap='gray')
plt.axis('off')

plt.subplot(133)
plt.title("Grayscale (Numpuy)")
plt.imshow(gray_numpy,cmap='gray')
plt.axis('off')

                                    
