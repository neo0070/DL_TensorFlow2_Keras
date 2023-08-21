""" 
    Deep Learning

    @file laboratory/pnuskgh/appl_vgg16.py
    @version 0.0.1
    @license OBCon License 1.0
    @copyright pnuskgh, All right reserved.
    @author gye hyun james kim <pnuskgh@gmail.com>
"""

import tensorflow as tf
from tensorflow import keras


#--- python laboratory/pnuskgh/appl_vgg16.py
class APPLICATION_VGG16():
    def __init__(self):
        self.name = 'vgg16'
        
    def load_data(self):
        pass

    def build_model(self, optimizer, loss_function, metric, modelType = 'default', allowLoad = True):
        model = None
        if (allowLoad):
            model = self.load_model()
        if (model != None):
            self.load_weights(model)
        else:
            if (modelType == 'default'):
                model = self.build_model_default()

        model.summary()
        model.compile(optimizer=optimizer, loss=loss_function, metrics=[ metric ])
        self.save_model(model)
        self.model = model
        
    def build_model_default(self):
        model = keras.applications.vgg16.VGG16(weights="imagenet", include_top=True)
        print(model)
        return model

if __name__ == "__main__":
    appl = APPLICATION_VGG16()
    appl.load_data()
    appl.build_model(keras.optimizers.Adam(), 'categorical_crossentropy', 'accuracy', 'default', False)
    # appl.process_model(20, 128, 1, 0.95, False)
