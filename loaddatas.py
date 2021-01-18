import numpy as np
import cv2
import os
import random
import pickle


class LoadData:
    pass

    class_names = ["live", "spoof"]
    data_set = "nuaa_dataset"

    images_array = []
    labels_array = []

    t_array = []

    image_width = 100
    image_height = 100

    def load_data(self):
        for cat in self.class_names:
            path_name = os.path.join(self.data_set, cat)
            labels = self.class_names.index(cat)
            f_path = os.listdir(path_name)

            for i in f_path:
                images = cv2.imread(os.path.join(path_name, i), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(images, (self.image_width, self.image_height))
                self.t_array.append([img, labels])

    def data_shuffle(self):
        self.load_data()
        random.shuffle(self.t_array)

    def convert_numpy(self):
        self.data_shuffle()

        for image, label in self.t_array:
            self.images_array.append(image)
            self.labels_array.append(label)

        self.images_array = np.array(self.images_array).reshape(-1, self.image_width)

    def save_loaded_data(self):
        self.convert_numpy()

        pickle_out = open("vgg16_model/Images.pickle", "wb")
        pickle.dump(self.images_array, pickle_out)
        pickle_out.close()

        pickle_out = open("vgg16_model/Labels.pickle", "wb")
        pickle.dump(self.labels_array, pickle_out)
        pickle_out.close()


load = LoadData()
load.save_loaded_data()
