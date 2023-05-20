import cv2
import pathlib
import logging
import numpy as np
import pandas as pd

from prepros import get_list_of_files

import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def gabor_filters_features(ksize, sigma, theta_range, lamda, gamma, phi):
    """Makes a Gabor filter bank with different with 8 filters
      with different orientation"""
    filters = []
    for theta in theta_range:
            kern = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lamda,
                gamma, phi, ktype=cv2.CV_64F
            )
            kern /= 1.0 * kern.sum()
            filters.append(kern)
    return filters


def apply_filter(processed_path_list, filters):
    """"""
    res = []
    depth = -1
    for img_path in processed_path_list:
        img = cv2.imread(img_path)
        newimage = np.zeros_like(img)
        flat_ar = []
        for kern in filters:
            image_filter = cv2.filter2D(img, depth, kern)
            np.maximum(newimage, image_filter, newimage)
            flat_ar.append(np.array(newimage))
        flat_ar.append(np.array(img))
        res.append(np.array(flat_ar).flatten())
        
    res = np.array(res)
    return res 


def find_optimal(img_files_list, df_target):
    """ Makes prediction, with different params of Gabor's filter. Writes the results
    (classification report, confusion matrix, train and test accuracy) in find_optimal.log.
    As a result gamma=0.2, sigma=1.1, lambda=3.5 gives the best accuracy, macro avg and 
    weighted avg """

    logging.basicConfig(level=logging.INFO, filename="find_optimal.log", filemode="w") 

    lamda_list = [3, 3.5, 4.0]  # Wavelength of sin component *pi/4
    theta_range = np.arange(0, np.pi, np.pi/8) # Orientation of the normal to the parallel stripes
    gamma_list = [0.2, 0.4, 0.6, 0.8, 1] # Spatial aspect ratio
    sigma_list = [0.5, 0.7, 0.9, 1.1] # Standart deviation
    phi = 0 # Phase
    ksize = 9 # Kernel size
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
                logging.info(classification(X_featured, df_target))


def classification(X_featured, df_target):
    """Splits data on train and test. The number of classes is uneven (there are too many neutral class),
    so we use upsampling for train data. We dont use upsampling for test data to avoid copied images
    that can influence result score. Than we use PCA to reduce data dimension and finaly LDA for classification"""
    y = np.array(df_target.label)
    X_train, X_test, y_train, y_test = train_test_split(X_featured, y, test_size=0.2)

    labeled_df = pd.DataFrame(data=X_train)
    labeled_df['label'] = y_train

    upsampled= pd.DataFrame(np.repeat(labeled_df[labeled_df.label != 0].values, 6, axis=0))
    upsampled.columns = labeled_df.columns
    upsampled_df = pd.concat([labeled_df[labeled_df.label == 0], upsampled])
    upsampled_df.label.value_counts()
    upsampled_df = upsampled_df.reset_index(drop=True)

    y_train = upsampled_df.label
    X_train = upsampled_df.drop('label', axis=1)

    print(X_featured.shape)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    scaler = StandardScaler() 
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    pca = PCA(0.95)
    pca.fit(X_train_scaled)
    x_train_pca = pca.transform(X_train_scaled)
    data_pca = pd.DataFrame(data=x_train_pca)
    X_train = data_pca

    scaler = StandardScaler()
    scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)
    x_test_pca = pca.transform(X_test_scaled)
    data_pca = pd.DataFrame(data=x_test_pca)
    X_test = data_pca

    LDA_model = LinearDiscriminantAnalysis()
    LDA_model.fit(X_train, y_train)
    LDA_prediction = LDA_model.predict(X_test)

    LDA_train_accuracy = LDA_model.score(X_train, y_train)
    LDA_test_accuracy = LDA_model.score(X_test, y_test)
    conf_matrix = confusion_matrix(LDA_prediction, y_test)
    report = classification_report(y_test, LDA_model.predict(X_test), output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    with open('pca.pkl', 'wb') as F:
        pickle.dump(pca, F)
    
    with open('lda_model.pkl', 'wb') as F:
        pickle.dump(LDA_model, F)

    return df_report, conf_matrix, LDA_train_accuracy, LDA_test_accuracy

if __name__ == "__main__":
    image_folder = pathlib.PurePath('F:/projects/vs_code/emoji_classification/processed')
    target_folder =  pathlib.PurePath('F:/projects/vs_code/emoji_classification/df_target.csv')
    img_files_list = get_list_of_files(image_folder)
    df_target = pd.read_csv(target_folder)

    find_optimal(img_files_list, df_target)
