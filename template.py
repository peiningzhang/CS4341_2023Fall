import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# MAGIC HAPPENS HERE
config = {
    'batch_size': 512,
    'image_size': (30,30),
    'epochs': 20,
    'optimizer': keras.optimizers.experimental.SGD(1e-2)
}
# MAGIC HAPPENS HERE

def read_data():
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        "./images/flower_photos",
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=config['image_size'],
        batch_size=config['batch_size'],
        labels='inferred',
        label_mode = 'int'
    )
    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)
    return train_ds, val_ds, test_ds

def data_processing(ds):
    data_augmentation = keras.Sequential(
        [
        #MAGIC HAPPENS HERE
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.3)

        #MAGIC HAPPENS HERE
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

    x = layers.Rescaling(1./255)(inputs)
    x = layers.Flatten()(x)
    #MAGIC HAPPENS HERE


    #MAGIC HAPPENS HERE
    outputs = layers.Dense(num_classes, activation="softmax", kernel_initializer='he_normal')(x)
    model = keras.Model(inputs, outputs)
    print(model.summary())
    return model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_ds, val_ds, test_ds = read_data()
    train_ds = data_processing(train_ds)
    model = build_model(config['image_size']+(3,), 5)
    model.compile(
        optimizer=config['optimizer'],
        loss='SparseCategoricalCrossentropy',
        metrics=["accuracy"],
    )
    history = model.fit(
        train_ds,
        epochs=config['epochs'],
        validation_data=val_ds
    )
    #MAGIC HAPPENS HERE
    print(history.history)


    #MAGIC HAPPENS HERE