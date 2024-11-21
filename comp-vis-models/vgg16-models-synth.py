import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow warnings
import numpy as np
import keras
from keras import layers, activations
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt

data_dir = # path to synthetic training dataset
test_data_dir = # path to synthetic dataset

batch_size = 32
img_height = 224 # default Resnet50 input size
img_width = 224

train_generator = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset='training',
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=969,
)

validation_generator = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset='validation',
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=969,
)

class_names = train_generator.class_names
print(class_names)

test_generator = keras.utils.image_dataset_from_directory(
    test_data_dir,
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=969
)

class_names = train_generator.class_names
print(class_names)
plt.figure(figsize=(10, 10))
for images, labels in train_generator.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[int(labels[i])])
        plt.axis('off')

plt.figure(figsize=(10, 10))
for images, labels in test_generator.take(10):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[int(labels[i])])
        plt.axis('off')
optimizers = ['adam', 'sgd']
activations = ['relu', 'silu', 'mish']

def sgd_optimizer():
    return keras.optimizers.SGD(learning_rate=1e-5)

def adam_optimizer():
    return keras.optimizers.Adam(learning_rate=1e-5)
csv_logger = CSVLogger('training_log_synth.csv', append=True)

epochs = 10

for activation in activations:
    for optimizer in optimizers:

        with open('training_log_synth.csv', 'a') as f:
            f.write(f'VGG16 - activation: {activation} optimizer: {optimizer}\n')
            f.write('epoch,train accuracy,train f1,train loss,train precision,train recall,val accuracy,val f1,val loss,val precision,val recall\n')

        keras.backend.clear_session()

        base_model = keras.applications.VGG16(
            weights='imagenet',
            input_shape=(img_height, img_width, 3),
            include_top=False,
            name='vgg16'
        )

        base_model.trainable = True

        inputs = keras.Input(shape=(img_height, img_width, 3))

        scale_layer = keras.layers.Rescaling(scale=1 / 255.0)
        x = scale_layer(inputs)

        x = base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(1, activation=activation)(x)

        model = keras.Model(inputs, outputs)

        model.summary(show_trainable=True)

        model.compile(
            optimizer=sgd_optimizer() if optimizer == 'sgd' else adam_optimizer(),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Precision(thresholds=0), keras.metrics.Recall(thresholds=0), 
                    keras.metrics.F1Score()],
        )

        print("Training model")
        history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[csv_logger])

        model.save(f'models/vgg16_synth_{activation}_{optimizer}.keras', overwrite=True)

        test_history = model.evaluate(test_generator, return_dict=True)

        with open('testing_log_synth.csv', 'a') as f:
            f.write(f'VGG16 - activation: {activation} optimizer: {optimizer}\n')
            f.write(f'binary_accuracy,loss,precision,recall,f1_score\n')
            f.write(f"{test_history['binary_accuracy']},{test_history['loss']},{test_history['precision']},{test_history['recall']},{test_history['f1_score']}\n")
            