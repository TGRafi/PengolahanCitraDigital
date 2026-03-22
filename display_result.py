import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import random

base_dir = os.path.dirname(__file__)
folder_name = 'dataset' 
path_to_dataset = os.path.join(base_dir, folder_name)

def load_and_process_dataset(path_folder):
    file_pattern = os.path.join(path_folder, "*.pgm")
    file_list = glob.glob(file_pattern)
    file_list.sort()
    
    if not file_list:
        print(f"File not found in: {path_folder}")
        return

    print(f"Succesfull find {len(file_list)} file .pgm!")
    
    # Ensure we have at least 5 samples; if not, use all available files
    num_samples = min(5, len(file_list))
    samples = random.sample(file_list, num_samples)
    plt.figure(figsize=(10, 10))
    
    for i, file_path in enumerate(samples):
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Skipping '{file_path}': unable to read image.")
                continue
            
            # 1. Intensity Transformation (Histogram Equalization)
            img_equ = cv2.equalizeHist(img)
            
            # 2. Spatial Filtering: Gaussian Blur (Smoothing)
            img_blur = cv2.GaussianBlur(img_equ, (5, 5), 0)

            # 3. Spatial Filtering: Laplacian (sharp/Edge Detection)
            img_laplacian = cv2.Laplacian(img_blur, cv2.CV_64F)
            img_laplacian = np.uint8(np.absolute(img_laplacian))

            display_list = [img, img_equ, img_blur, img_laplacian]
            titles = ['Original', 'Equalized', 'Gaussian Blur', 'Laplacian']
            
            for j in range(4):
                plt.subplot(5, 4, i*4 + j + 1)
                plt.imshow(display_list[j], cmap='gray')
                if i == 0: plt.title(titles[j])
                plt.axis('off')
        except cv2.error as err:
            print(f"OpenCV error while processing '{file_path}': {err}")

    plt.tight_layout()
    plt.show()

load_and_process_dataset(path_to_dataset)