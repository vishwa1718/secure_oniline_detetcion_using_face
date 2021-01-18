import pickle

import matplotlib.pyplot as plt
import tensorflow as tf


class VGG16CNN:
    pass

    epoch_size = 25
    b_size = 34
    num_split = 0.2
    file_name = 'vgg16_model/CNN_25.h5'
    image_width = 100
    image_height = 100

    img = pickle.load(open("vgg16_model/Images.pickle", "rb"))
    lbl = pickle.load(open("vgg16_model/Labels.pickle", "rb"))

    IMG = img/225.0

    def create_vgg16_cnn(self):

        forward_model = tf.keras.Sequential()

        forward_model.add(tf.keras.layers.Conv2D(8, (3, 3), input_shape=(self.image_width, self.image_height, 1)))
        forward_model.add(tf.keras.layers.Activation('relu'))
        forward_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        forward_model.add(tf.keras.layers.Conv2D(16, (3, 3)))
        forward_model.add(tf.keras.layers.Activation('relu'))
        forward_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        forward_model.add(tf.keras.layers.Conv2D(32, (3, 3)))
        forward_model.add(tf.keras.layers.BatchNormalization())
        forward_model.add(tf.keras.layers.Activation('relu'))
        forward_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        forward_model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        forward_model.add(tf.keras.layers.BatchNormalization())
        forward_model.add(tf.keras.layers.Activation('relu'))
        forward_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        forward_model.add(tf.keras.layers.Flatten())
        forward_model.add(tf.keras.layers.Dense(64))
        forward_model.add(tf.keras.layers.Activation('relu'))

        forward_model.add(tf.keras.layers.Dense(1))
        forward_model.add(tf.keras.layers.Activation('sigmoid'))
        forward_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = forward_model.fit(self.IMG, self.lbl, batch_size=self.b_size, epochs=self.epoch_size,
                                    validation_split=self.num_split)

        fig = plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        fig.savefig("vgg16_model/dataset1_accuracy25.png", dpi=fig.dpi)

        fig2 = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        fig2.savefig('vgg16_model/dataset1_loss25.png', dpi=fig2.dpi)


        model_json = forward_model.to_json()
        with open("vgg16_model/model_25.json", "w") as json_file:
            json_file.write(model_json)
        forward_model.save_weights(self.file_name)
        print("Saved model to disk")


v = VGG16CNN()
v.create_vgg16_cnn()
