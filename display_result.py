import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import random

base_dir = os.path.dirname(__file__)
folder_name = 'dataset'
path_to_dataset = os.path.join(base_dir, folder_name)


def gamma_correction(img, gamma=1.0):
    # buat lookup table untuk semua 256 nilai piksel yang mungkin
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(img, table)



def load_and_process_dataset(path_folder):
    file_pattern = os.path.join(path_folder, "*.pgm")
    file_list = glob.glob(file_pattern)
    file_list.sort()

    if not file_list:
        print(f"File not found in: {path_folder}")
        return

    print(f"Succesfull find {len(file_list)} file .pgm!")

    # pastikan minimal 5 sampel; jika kurang, gunakan semua file yang ada
    num_samples = min(5, len(file_list))
    samples = random.sample(file_list, num_samples)

    num_cols = 4
    plt.figure(figsize=(18, 10))

    for i, file_path in enumerate(samples):
        try:
            # muat gambar dalam skala abu-abu
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Skipping '{file_path}': unable to read image.")
                continue

            # 1. transformasi intensitas - histogram equalization
            img_equ = cv2.equalizeHist(img)

            # 2. filter spasial - gaussian blur (penghalusan)
            img_blur = cv2.GaussianBlur(img_equ, (5, 5), 0)

            # 3. filter spasial - laplacian (deteksi tepi)
            img_laplacian = cv2.Laplacian(img_blur, cv2.CV_64F)
            img_laplacian = np.uint8(np.absolute(img_laplacian))

            # 4. koreksi gamma (power-law) - gamma terang
            img_gamma_bright = gamma_correction(img, gamma=0.45)


            display_list = [
                img, img_equ, img_blur, img_laplacian
                , img_gamma_bright
            ]
            titles = [
                'original', 'equalized', 'gaussian blur', 'laplacian',
                'gamma gelap'
            ]

            for j in range(num_cols + 1):
                plt.subplot(num_samples, num_cols + 1, i * (num_cols + 1) + j + 1)
                plt.imshow(display_list[j], cmap='gray')
                if i == 0:
                    plt.title(titles[j], fontsize=7)
                plt.axis('off')

        except cv2.error as err:
            print(f"OpenCV error while processing '{file_path}': {err}")

    plt.tight_layout()
    plt.show()


load_and_process_dataset(path_to_dataset)