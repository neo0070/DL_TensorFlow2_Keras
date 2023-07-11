""" 
    Deep Learning with Tensorflow 2 and Keras

    @file laboratory/pnuskgh/chapter_1_065.py
    @version 0.0.1
    @license OBCon License 1.0
    @copyright pnuskgh, All right reserved.
    @author gye hyun james kim <pnuskgh@gmail.com>
"""

import tensorflow as tf
from tensorflow import keras

from chapter_1_065 import TensorFlow2_065

class TensorFlow2_073(TensorFlow2_065):
    def __init__(self):
        super().__init__()
        
    def initialize(self):
        super().initialize()
        
        self.epochs = 50                                                        #--- 훈련 집합 횟수
        
    def build_model(self):
        reshaped = 28 * 28                                                      #--- 행열(28 * 28)을 벡터(784 * 1)로 변환
        n_hidden = 128
        nb_classes = 10                                                         #--- 분류 갯수

        model = tf.keras.models.Sequential()                                    #--- 모델 : Sequential
        model.add(keras.layers.Dense(n_hidden, input_shape=(reshaped,),         #--- 출력 갯수, 입력 갯수
                name="dense_layer_1", activation='relu'                         #--- Activation Function
        ))
        model.add(keras.layers.Dense(n_hidden,                                  #--- 출력 갯수
                name="dense_layer_2", activation='relu'                         #--- Activation Function
        ))
        model.add(keras.layers.Dense(nb_classes,                                #--- 출력 갯수
                name="dense_layer_3", activation=self.activation_function       #--- Activation Function
        ))

        model.summary()
        model.compile(
            optimizer=self.optimizer,                                           #--- Optimizer
            loss=self.loss_function,                                            #--- Loss Function
            metrics=[ self.metrics ],                                           #--- Matric
        )
        return model

if __name__ == "__main__":
    deep_learning = TensorFlow2_073()
    deep_learning.initialize()
    
    (x_train, y_train), (x_test, y_test) = deep_learning.load_data()
    model = deep_learning.build_model()
    deep_learning.process_model(model, x_train, y_train, x_test, y_test)
