import numpy as np
import cv2
from pathlib import Path
from pickle import load
from preprocessing import image_pre_processing
from training import gabor_filters_features, apply_filter



def apply_model(image_folder, models_folder):
    """Applying trained model to any picture and returns classified emotion"""

    img_processed = image_pre_processing(image_folder, 1, 64, 64)
    processed_path = str(image_folder[0].parent.joinpath("prosecced.jpg"))
    cv2.imwrite(processed_path, img_processed[0])

    theta_range = np.arange(0, np.pi, np.pi/8)
    gamma = 0.2
    sigma = 1.1
    ksize = 9
    phi = 0
    lamda = 3.5 * np.pi/4

    filters = gabor_filters_features(ksize, sigma, theta_range,
                                     lamda, gamma, phi)
    
    img_features = apply_filter([processed_path], filters)


    with open(models_folder.joinpath('scaler.pkl'), 'rb') as file:
        scaler = load(file)

    img_features = scaler.transform(img_features)

    with open(models_folder.joinpath('pca.pkl'), 'rb') as file:
        pca = load(file)
    
    img_features = pca.transform(img_features)

    with open(models_folder.joinpath('lda_model.pkl'), 'rb') as file:
        lda = load(file)

    img_features = lda.transform(img_features)

    with open(models_folder.joinpath('svd_model.pkl'), 'rb') as file:
        svd = load(file)

    predict = svd.predict(img_features)

    dict_emotion = {
        0:'neutral',
        1:'anger',
        2:'contempt',
        3:'disgust',
        4:'fear',
        5:'happiness',
        6:'surprise'
    }
    emotion = dict_emotion[predict[0]]
    print(emotion)

    return emotion


if __name__ == "__main__":
    project_folder = Path(__file__).parents[1].resolve()
    image_folder = list(project_folder.joinpath('data/to_predict/04.jpg'))
    models_folder = project_folder.joinpath('models')

    apply_model(image_folder, models_folder)

    
