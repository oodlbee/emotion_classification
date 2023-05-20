from pathlib import Path
import logging
import pandas as pd
import numpy as np
from training import gabor_filters_features, lda_classification, apply_filter
from preprocessing import get_list_of_files


def find_optimal(img_files_list, df_target):
    """ Makes prediction, with different params of Gabor's filter. Writes the results
    (classification report, confusion matrix, train and test accuracy) in 
    find_optimal.log. As a result gamma = 0.2, sigma=1.1, lambda=3.5 gives
    the best accuracy, macro avg and weighted avg """

    logging.basicConfig(level=logging.INFO, filename="find_optimal.log", filemode="w") 
    lamda_list = [3, 3.5, 4.0]  # Wavelength of sin component *pi/4
    theta_range = np.arange(0, np.pi, np.pi/8) # Orientation of the normal to the parallel stripes
    gamma_list = [0.2, 0.4, 0.6, 0.8, 1] # Spatial aspect ratio
    sigma_list = [0.5, 0.7, 0.9, 1.1] # Standart deviation

    lamda_list = [3, 4.0]  # Wavelength of sin component *pi/4
    theta_range = np.arange(0, np.pi, np.pi/8) # Orientation of the normal to the parallel stripes
    gamma_list = [0.2, 0.6, 1] # Spatial aspect ratio
    sigma_list = [0.5, 1.1] # Standart deviation
    phi = 0 # Phase
    ksize = [5, 9, 15]# Kernel size
    i = 0
    for gamma in gamma_list:
        for sigma in sigma_list:
            for  lamda_val in lamda_list:
                i += 1
                lamda = lamda_val * np.pi/4
                filters = gabor_filters_features(ksize, sigma, theta_range, lamda, gamma, phi)
                X_featured = apply_filter(img_files_list, filters)
                logging.info(f'{i} itteration')
                logging.info(f'gamma = {gamma}, sigma = {sigma}, lambda = {lamda_val} * pi/4')
                logging.info("Result: ")
                logging.info(lda_classification(X_featured, df_target))


if __name__ == "__main__":

    project_folder = Path().resolve()
    image_folder = project_folder.joinpath('data/processed/images')
    target_path = project_folder.joinpath('data/processed/df_target.csv')
    img_files_list = get_list_of_files(image_folder)
    df_target = pd.read_csv(target_path)

    find_optimal(img_files_list, df_target)
