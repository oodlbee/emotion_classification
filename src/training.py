import pandas as pd
import numpy as np
import pathlib

from gabor_bank import gabor_filters_features, apply_filter, classification
from prepros import get_list_of_files




if __name__ == "__main__":
    image_folder = pathlib.PurePath('F:/projects/vs_code/emoji_classification/processed')
    target_folder =  pathlib.PurePath('F:/projects/vs_code/emoji_classification/df_target.csv')
    img_files_list = get_list_of_files(image_folder)
    df_target = pd.read_csv(target_folder)

    theta_range = np.arange(0, np.pi, np.pi/8) 
    gamma = 0.2 
    sigma = 1.1 
    ksize = 9
    phi = 0 
    lamda = 3.5 * np.pi/4

    filters = gabor_filters_features(ksize, sigma, theta_range, lamda, gamma, phi)
    X_featured = apply_filter(img_files_list, filters)
    print(classification(X_featured, df_target))