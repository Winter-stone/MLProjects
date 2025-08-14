import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def is_image_corrupted(image_path):
    try:
        img = Image.open(image_path)
        return False
    except (IOError, SyntaxError) as e:
        print(f"Corrupted image found: {image_path}")
        return True


def delete_or_keep():
    categories = ['dandelion', 'daisy', 'tulip', 'sunflower', 'rose']
    for i in categories:
        folder_path = f'dataset\Flowers/{i}'

        clean_images = []
        total_images = []
        corrupted_images = []

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            if os.path.isfile(image_path):
                total_images.append(image_path)
                if is_image_corrupted(image_path):
                    corrupted_images.append(image_path)

        for image_path in corrupted_images:
            os.remove(image_path)
            print(f"Deleted corrupted image: {image_path}")

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            if os.path.isfile(image_path):
                clean_images.append(image_path)


        print(f'Total images of {i} before Cleansing: ', len(total_images))
        print(f'Total images of {i} after Cleansing: ', len(clean_images))
        print(f"Total corrupted images of {i} found: {len(corrupted_images)}")


def train_test_data():
    datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2, rotation_range=10,
                             width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                             fill_mode ='nearest')

    train_generator = datagen.flow_from_directory('dataset\Flowers', target_size=(224,224),
                                                  batch_size=8, class_mode='categorical', subset='training')

    validation_generator = datagen.flow_from_directory('dataset\Flowers', target_size=(224,224),
                                                  batch_size=8, class_mode='categorical', subset='validation')

    return train_generator, validation_generator

if __name__ == '__main__':
    delete_or_keep()
    train_test_data()