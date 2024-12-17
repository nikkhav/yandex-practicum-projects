import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

def load_train(path):
    labels_csv = f"{path}/labels.csv"
    images_dir = f"{path}/final_files"
    df = pd.read_csv(labels_csv)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.25,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory=images_dir,
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=42
    )
    return train_generator

def load_test(path):
    labels_csv = f"{path}/labels.csv"
    images_dir = f"{path}/final_files"
    df = pd.read_csv(labels_csv)

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.25
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df,
        directory=images_dir,
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=42
    )
    return test_generator

def create_model(input_shape=(150, 150, 3)):
    backbone = ResNet50(
        input_shape=input_shape,
        weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        include_top=False
    )

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])
    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=20, steps_per_epoch=None, validation_steps=None):
    if steps_per_epoch is None:
        steps_per_epoch = train_data.samples // train_data.batch_size
    if validation_steps is None:
        validation_steps = test_data.samples // test_data.batch_size

    model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
        shuffle=True
    )
    return model
