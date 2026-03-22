import cv2
import numpy as np
import os
import glob

base_dir = os.path.dirname(__file__)
folder_name = 'dataset'
path_to_dataset = os.path.join(base_dir, folder_name)

folder_output = 'dataset_hasil'
path_output = os.path.join(base_dir, folder_output)

if not os.path.exists(path_output):
    os.makedirs(path_output)
    print(f"Folder '{folder_output}' created successfully in {base_dir}")
else:
    print(f"Folder '{folder_output}' already available. New images will be added in it.")

def load_process_save_dataset(path_folder):
    file_pattern = os.path.join(path_folder, "*.pgm")
    file_list = glob.glob(file_pattern)
    file_list.sort()
    
    if not file_list:
        print(f"File not found in: {path_folder}")
        return

    print(f"Succesfull find {len(file_list)} file .pgm! Processing and saving...")

    processed_count = 0

    for i, file_path in enumerate(file_list, start=1):
        try:
            # 1. Load Gambar (Original)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Skipping '{file_path}': unable to read image.")
                continue

            # 2. Intensity Transformation (Histogram Equalization)
            img_equ = cv2.equalizeHist(img)

            # 3. Spatial Filtering: Gaussian Blur (Smoothing)
            img_blur = cv2.GaussianBlur(img_equ, (5, 5), 0)

            # 4. Spatial Filtering: Laplacian (Edge Detection)
            img_laplacian = cv2.Laplacian(img_equ, cv2.CV_64F)
            img_laplacian = np.uint8(np.absolute(img_laplacian))

            # 5. Get File Name Without Extension
            base_filename = os.path.basename(file_path)
            filename_without_ext = os.path.splitext(base_filename)[0]

            # 6. Specify the Full Path to Save the Results
            output_path_equ = os.path.join(path_output, filename_without_ext + '_equ.pgm')
            output_path_blur = os.path.join(path_output, filename_without_ext + '_blur.pgm')
            output_path_laplacian = os.path.join(path_output, filename_without_ext + '_laplacian.pgm')

            # 7. Save Image to Output Folder
            cv2.imwrite(output_path_equ, img_equ)
            cv2.imwrite(output_path_blur, img_blur)
            cv2.imwrite(output_path_laplacian, img_laplacian)
            processed_count += 1

            if i % 10 == 0:
                print(f"{i} images have been successfully processed so far.")

        except cv2.error as err:
            print(f"OpenCV error while processing '{file_path}': {err}")

    print(f"All processed images have been saved in the folder: {folder_output}. Total saved: {processed_count}")

load_process_save_dataset(path_to_dataset)