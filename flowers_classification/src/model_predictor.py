import pickle as pkl
from io import BytesIO
import requests
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from requests.exceptions import InvalidSchema, MissingSchema

loaded_model = pkl.load(open('model_flowers.keras', 'rb'))
labels = loaded_model['class_name']
loaded_model = loaded_model['model']

################################## Loading a Image from a web browser URL #####################################

img_url = "https://images.contentstack.io/v3/assets/bltcedd8dbd5891265b/blt458318b3c95ee295/6668c752494a74655c257a1a/purple-tulips-blooming.jpg?q=70&width=3840&auto=webp"
if img_url:
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).resize((224, 224))
        print(response.status_code)

    except MissingSchema as e:
        print("Missing URL schema error:", e)
        print('Try a Different URL: http(s)/...')

    except InvalidSchema as e:
        print("Invalid URL schema error:", e)
        print('Try a Different URL: http(s)/...')

    except UnidentifiedImageError as e:
        print("Invalid URL schema error:", e)
        print('URL not found, Try a Different URL / http(s)/...')

    else:
        image = np.reshape(img, [1, 224, 224, 3])
        image = image / 255
        prediction = loaded_model.predict(image)
        prediction_index = np.argmax(prediction[0])

        print(prediction)
        print(f'This Picture is of a {labels[prediction_index]} with {round(np.max(prediction), 2) * 100}% certainty')
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')  # Hide the axes
        plt.show()

################################## Loading a Image from a local directory #####################################

else:
    try:
        image = tf.keras.preprocessing.image.load_img('dataset\Flowers/daisy/104.jpg')

    except FileNotFoundError as e:
        print('File Not Found at', e)
        print('Ensure that the file path is correct')

    else:
        img = image.resize((224, 224))
        image = np.reshape(img, [1, 224, 224, 3])
        image = image / 255
        prediction = loaded_model.predict(image)
        prediction_index = np.argmax(prediction[0])

        print(prediction)
        print(f'This Picture is of a {labels[prediction_index]} with {round(np.max(prediction * 100), 2)}% certainty')
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()