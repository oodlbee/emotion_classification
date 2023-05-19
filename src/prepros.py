import os
import cv2
import pathlib
import numpy as np
import pandas as pd
import face_recognition as fr


def get_list_of_files(dirName):
    """Makes a list of strings of file's paths from directiory"""
    listOfFile = os.listdir(dirName)            # Сделать через pathlib
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_list_of_files(fullPath)
        else:
            allFiles.append(fullPath)             
    return allFiles


def image_pre_processing(data, sigma_GF, IMG_HEIGHT, IMG_WIDTH):
    """Applies Gaussian filter to remove noise, applies histogram 
    equalisation and scales image to convert all data to a common format"""
    X = []
    for image in data: 
        image = cv2.imread(image)

        image_GF = cv2.GaussianBlur(image, (5, 5), sigma_GF)    # Gaussian filter

        face_loc = fr.face_locations(img = image_GF, model='hog')   # Face detection and crop
        if len(face_loc) > 1:
          face_loc=[(face_loc[0])]
        top, right, bottom, left = face_loc[0]
        image_CROPPED = image_GF[top:bottom, left:right]

        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))   # Histogram equalization
        lab = cv2.cvtColor(image_CROPPED, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2,a,b))
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)

        image_SCALED = cv2.resize(img_clahe, (IMG_HEIGHT, IMG_WIDTH))
        X.append(image_SCALED)

    X = np.array(X)
    return X 


def save_images(images_array, save_path, target_df):
    """Saves processed images to directory"""
    try: 
        os.mkdir(save_path) 
    except OSError as er: 
        pass 
    for i, image in enumerate(images_array):
        name = target_df.iloc[i]['name']
        path = str(save_path.joinpath(name))
        cv2.imwrite(path, image)


def make_target_df(target_folder, files_list):
    """Makes from dataset description target csv file"""
    df_target = pd.read_csv(target_folder) 
    df_target = df_target.drop(columns=['Unnamed: 0'], axis=1)
    label_list = []
    name_list = []
    for file in files_list:
        pic_name = file.split('\\')[-1]                    # Сделать через pathlib       
        idx = df_target[df_target['file name'] == pic_name].index
        label_list.append(df_target.at[idx[0], 'label'])
        name_list.append(pic_name)
    dic = {'name':name_list,'label':label_list}
    df_target = pd.DataFrame(dic)
    path = target_folder.parents[0].joinpath('df_target.csv')
    df_target.to_csv(path)

    return df_target


if __name__ == '__main__':
    read_folder = pathlib.PurePath('F:/projects/vs_code/emoji_classification/all')
    target_folder = pathlib.PurePath('F:/projects/vs_code/emoji_classification/label_dataset.csv')

    img_files_list = get_list_of_files(read_folder)

    target_df = make_target_df(target_folder, img_files_list)
    X_processed = image_pre_processing(img_files_list, 1, 64, 64)

    write_folder = read_folder.parents[0].joinpath('processed')
    save_images(X_processed, write_folder, target_df)