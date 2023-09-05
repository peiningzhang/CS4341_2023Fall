# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
def read_data():
    for file_name in os.listdir('./images/images'):
        print(file_name)

    image_size = (180, 180)
    batch_size = 64
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        "./images/images",
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        labels='inferred',
        label_mode = 'int'
    )

    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(int(labels[i]))
    #         plt.axis("off")
    return train_ds, val_ds


def data_processing(ds):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.3),
        ]
    )

    ds = ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
def build_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(scale=1./127.5, offset=-1)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 768]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    model = keras.Model(inputs, outputs)

    # keras.utils.plot_model(model, show_shapes=True)
    return model

def train_model(model, train_ds, val_ds):
    epochs = 100

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='SparseCategoricalCrossentropy',
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_ds, val_ds = read_data()
    train_ds = data_processing(train_ds)
    # val_ds = val_ds
    model = build_model((180, 180)+(3,), 5)
    train_model(model, train_ds, val_ds)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
