import pickle as pkl
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model

from data_ingestion import train_test_data

train_data, val_data = train_test_data()

base_model = InceptionV3(input_shape=(224,224,3), include_top=False, weights = 'imagenet')
for layer in base_model.layers:
  layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
preds = Dense(5, activation = 'softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, epochs=5, validation_data=val_data)

for layer in base_model.layers[-30:]:
  layer.trainable = True
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=5, validation_data=val_data)
test_loss, test_acc = model.evaluate(val_data)

class_names = {v:k for k,v in train_data.class_indices.items()}
print('classes',class_names)
data_model = {'class_name': class_names, 'model':model}
pkl.dump(data_model, open('model_flowers.keras', 'wb'))