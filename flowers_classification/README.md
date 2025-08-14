# Pre-trained Image Classification Model

### Transfer learning with google's Inception V3 model in python.

#### This pre-trained image classification model is a machine learning program that is trained on 5 different flowers [dandelion, daisy, tulips, sunflowers, roses] making it a categorical classification. It then predicts what the image inputted by the user is with a certainty score in %.

### Requirements 

#### This program uses frameworks such as Tensorflow, keras, and libraries such as Pillow, pandas, etc.. if you have not installed the packages, you can install them as shown below on your local machine terminal.

pip install pandas as pd
pip install numpy as np
pip install matplotlib
pip install tensorflow as tf
pip install scikit-learn
pip install Pillow
pip install requests

#### Alternatively you could run the setup.py file, and every package will be installed by using the terminal: pip install -r \folder path\requirements.txt

### Train Model

#### Inside the src file there is a python file named data_ingestion.py, its main function is to go through the dataset of images, check for corrupted images and delete if found, and generate every other image the split them into train_set and validation_set.

#### Data is prepared, now running the file model_trainer.py will train the model on available data using the Inception V3 pre-trained model as the base model.

### Model Prediction

#### The model is then run for predictions using the model_predictor.py file to input an image and wait for the result of what flower the image is with a certainty score in %. There's one of two ways the user can input an image for prediction, either from a URL, or from the local machine.


## End of program