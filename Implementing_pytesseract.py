import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Reading the image into python.
image_path = r'C:\Users\cheta\Downloads\d.png'
img = cv2.imread(image_path)
# cv2.imshow("Image", img)

# #Inverting an image
# inverted_image = cv2.bitwise_not(img)
# cv2.imshow("INVERTED", inverted_image)


# Grayscaling of an image
gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("GRAYSCALE",gray_scale)

# Binazriation using thresholding
# Various methods of thresholding
ret, thresh1 = cv2.threshold(gray_scale, 170, 255, cv2.THRESH_BINARY_INV)
# ret, thresh2 = cv2.threshold(gray_scale, 120, 255, cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(gray_scale, 120, 255, cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(gray_scale, 120, 255, cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(gray_scale, 120, 255, cv2.THRESH_TOZERO_INV)

cv2.imshow("Binarized", thresh1)
# cv2.imshow("Binarized_inverted", thresh2)
# cv2.imshow("Truncated_image", thresh3)
# cv2.imshow("To_zero", thresh4)
# cv2.imshow("To_zero_inverted", thresh5)

# Denoising the image
denoised_image = cv2.fastNlMeansDenoising(thresh1, dst=None, h=10, templateWindowSize=None, searchWindowSize=None)
cv2.imshow("denoised_image", denoised_image)

# #Dilation and Erosion of the image
# kernel = np.ones((3, 3), np.uint8)
# img_erosion = cv2.erode(thresh1, kernel, iterations=1)
# img_dilate = cv2.dilate(thresh1, kernel, iterations=1)
# cv2.imshow("Eroded_image", img_erosion)
# cv2.imshow("Dilated_image", img_dilate)

# Implementing the pytesseract engine to find texts in images after preprocessing.
text = pytesseract.image_to_string(img)

print(text)

cv2.waitKey(0)

# ALL THE ERRORS ENCOUNTERED
# img = cv2.imread('C:\Users\cheta\Downloads\ocr1.jpg')
# This error occurs because you are using a normal string as a path.

