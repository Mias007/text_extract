import cv2
import numpy as np
import pytesseract
from django.conf import settings
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"


def spatial_processing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoise = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(denoise, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 3)
    return thresh


def frequency_processing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2

    mask = np.ones((rows, cols), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 0

    filtered = fshift * mask
    ishift = np.fft.ifftshift(filtered)
    img_back = np.abs(np.fft.ifft2(ishift))

    img_back = cv2.normalize(img_back, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
    return img_back


def extract_text(img):
    return pytesseract.image_to_string(img)
