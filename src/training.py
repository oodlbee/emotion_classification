import cv2
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from preprocessing import get_list_of_files


def gabor_filters_features(ksize, sigma, theta_range, lamda, gamma, phi):
    """Makes a Gabor filter bank with different with 8 filters
      with different orientation"""
    filters = []
    for theta in theta_range:
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma,
                                  phi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()
        filters.append(kern)
    return filters


def apply_filter(processed_path_list, filters):
    """Applies gabor bank to the image. One image converts to eight."""
    res = []
    depth = -1
    for img_path in processed_path_list:
        img = cv2.imread(str(img_path))
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


def lda_classification(X_featured, df_target, write_folder):
    """Splits data on train and test. The number of classes is uneven
    (there are too many neutral class),so we use upsampling for train data.
    We dont use upsampling for test data to avoid copied images that can
    influence result score. Than we use PCA + LDA to reduce data dimension and finaly
    SVM for classification. During experiments in notebook 
    (/notebooks/emotion_classification.ipybn), SVM showed the best results in classification"""

    y = np.array(df_target.label)
    X_train, X_test, y_train, y_test = train_test_split(X_featured, y,
                                                        test_size=0.2)

    # upsampling
    labeled_df = pd.DataFrame(data=X_train)
    labeled_df['label'] = y_train

    upsample = pd.DataFrame(np.repeat(
        labeled_df[labeled_df.label != 0].values, 3, axis=0))
    upsample.columns = labeled_df.columns
    resampled_df = pd.concat([labeled_df[labeled_df.label == 0], upsample], ignore_index=True)

    # downsample = resampled_df[resampled_df.label == 0]
    # drop_indices = np.random.choice(downsample.index, int(len(downsample)/2), replace=False)
    # resampled_df = resampled_df.drop(drop_indices)

    resampled_df = resampled_df.reset_index(drop=True)    

    y_train = resampled_df.label
    X_train = resampled_df.drop('label', axis=1)

    # PCA + LDA to dimensionality reduction
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    pca = PCA(0.95)
    pca.fit(X_train_scaled)
    x_train_pca = pca.transform(X_train_scaled)
    X_train = pd.DataFrame(data=x_train_pca)

    scaler = StandardScaler()
    scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)
    x_test_pca = pca.transform(X_test_scaled)
    X_test = pd.DataFrame(data=x_test_pca)

    LDA_model = LinearDiscriminantAnalysis()
    LDA_model.fit(X_train, y_train)

    x_train_lda = LDA_model.transform(X_train)
    X_train = pd.DataFrame(data=x_train_lda)
    x_test_lda = LDA_model.transform(x_test_pca)
    X_test = pd.DataFrame(data=x_test_lda)

    # SVM classification
    clf = SVC()
    params = {'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.001, 0.0001], 'degree': range(1, 15), 'kernel': ['linear', 'rbf']}
    search = GridSearchCV(clf, params, scoring="f1_weighted", cv=5, n_jobs=-1)
    search.fit(x_train_lda, y_train)

    SVC_model = search.best_estimator_
    SVC_model.fit(x_train_lda, y_train)
    SVC_prediction = SVC_model.predict(x_test_lda) 

    report = classification_report(y_test, SVC_prediction, output_dict=True)
    conf_matrix = confusion_matrix(SVC_prediction, y_test)
    df_conf_matrix = pd.DataFrame(conf_matrix)
    df_report = pd.DataFrame(report).transpose()


    with open(write_folder.joinpath('pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)

    with open(write_folder.joinpath('lda_model.pkl'), 'wb') as f:
        pickle.dump(LDA_model, f)

    with open(write_folder.joinpath('svd_model.pkl'), 'wb') as f:
        pickle.dump(SVC_model, f)

    df_report.to_excel(write_folder.joinpath('classification_report.xlsx'))
    df_conf_matrix.to_excel(write_folder.joinpath('confusion_matrix.xlsx'))


if __name__ == "__main__":

    project_folder = Path(__file__).parents[1].resolve()
    image_folder = project_folder.joinpath('data/processed/images')
    target_path = project_folder.joinpath('data/processed/df_target.csv')
    write_folder = project_folder.joinpath('models')


    img_files_list = get_list_of_files(image_folder)
    df_target = pd.read_csv(target_path)

    theta_range = np.arange(0, np.pi, np.pi/8)
    gamma = 0.2
    sigma = 1.1
    ksize = 9
    phi = 0
    lamda = 3.5 * np.pi/4

    filters = gabor_filters_features(ksize, sigma, theta_range,
                                     lamda, gamma, phi)
    X_featured = apply_filter(img_files_list, filters)

    lda_classification(X_featured, df_target, write_folder)
