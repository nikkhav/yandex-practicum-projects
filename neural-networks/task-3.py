from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

def load_train(path):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8,1.2],
        validation_split=0.25
    )
    train_generator = train_datagen.flow_from_directory(
        path,
        target_size=(150,150),
        batch_size=32,
        class_mode='sparse',
        subset='training',
        seed=42
    )
    return train_generator

def create_model(input_shape=(150,150,3), num_classes=12):
    model = Sequential()
    model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=20, steps_per_epoch=None, validation_steps=None):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)
    model.fit(train_data,validation_data=test_data,batch_size=batch_size,epochs=epochs,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,verbose=2,shuffle=True)
    return model
