import cv2
from pathlib import Path
import numpy as np
import pandas as pd
import face_recognition as fr


def get_list_of_files(dir_path):
    """Makes a list of strings of file's paths from directiory"""
    list_of_file = Path.iterdir(dir_path)           
    all_files = []
    for file in list_of_file:
        full_path = dir_path.joinpath(file)
        if Path.is_dir(full_path):
            all_files = all_files + get_list_of_files(full_path)
        else:
            all_files.append(full_path)             
    return all_files


def image_pre_processing(data, sigma_GF, IMG_HEIGHT, IMG_WIDTH):
    """Applies Gaussian filter to remove noise, applies histogram 
    equalisation and scales image to convert all data to a common format"""
    X = []
    for path in data:
        image = cv2.imread(str(path))

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
        Path.mkdir(save_path) 
    except OSError as er: 
        pass 
    for i, image in enumerate(images_array):
        name = target_df.iloc[i]['name']
        path = str(save_path.joinpath(name))
        cv2.imwrite(path, image)
    target_df.to_csv(save_path.parent.joinpath('df_target.csv'))


def make_target_df(target_folder, files_list):
    """Makes from dataset description target csv file"""
    df_target = pd.read_csv(target_folder) 
    df_target = df_target.drop(columns=['Unnamed: 0'], axis=1)
    
    label_list = []
    name_list = []
    for file in files_list:
        pic_name = file.name      
        idx = df_target[df_target['file name'] == pic_name].index
        label_list.append(df_target.at[idx[0], 'label'])
        name_list.append(pic_name)
    dic = {'name':name_list,'label':label_list}
    df_target = pd.DataFrame(dic)
    return df_target


if __name__ == '__main__':
    project_folder = Path(__file__).parents[1].resolve()
    read_folder = project_folder.joinpath('data/raw/images')
    target_path = project_folder.joinpath('data/raw/label_dataset.csv')
    write_folder = project_folder.joinpath('data/processed/images')
    
    img_files_list = get_list_of_files(read_folder)
    target_df = make_target_df(target_path, img_files_list)
    X_processed = image_pre_processing(img_files_list, 1, 64, 64)
    
    save_images(X_processed, write_folder, target_df)