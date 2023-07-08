# coding=utf-8
#
# @file laboratory/pnuskgh/chapter_1_001.py
# @version 0.0.1
# @license OBCon License 1.0
# @copyright pnuskgh, All right reserved.
# @author gye hyun james kim <pnuskgh@gmail.com>

# --- https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=mnist
# --- python  laboratory/pnuskgh/chapter_1_065.py

import os
import tensorflow as tf
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # --- 0. 0번 GPU 사용, -1. GPU 사용하지 않음

print(f"TensorFlow version: {tf.__version__}")

# --- 데이터 수집과 가공
RESHAPED = 28 * 28  # --- 행열(28 * 28)을 벡터(784 * 1)로 변환
NB_CLASSES = 10  # --- 분류 갯수

mnist = keras.datasets.mnist
(X_train, Y_train), (
    X_test,
    Y_test,
) = mnist.load_data()  # --- 60,000/10,000개 데이터 (28 * 28)

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train /= 255  # --- Normalize
X_test /= 255
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

# --- 모델링
# ---     Models : Sequential (순차), Functional (함수), Model Subclassing
# ---     Activation Functions (활성화 함수) : Sigmoid ([0, 1]), TanH ([-1, 1]), ReLU ([0, x]), ELU (TanH + ReLU), LeakyReLu
# ---     Loss Functions (손실 함수) : categorical_crossentropy, MSE, binary_crossentropy
# ---     Optimizers (최적화) : SGD (Stochastic Gradient Descent, 확률적 그래디언트 하강)
# ---     Metrics (척도) : Accuracy, Precision, Recall
EPOCHS = 200  # --- 전체 훈련 횟수
BATCH_SIZE = 128  # --- 각 훈련당 표본의 크기
VERBOSE = 1
NB_CLASSES = 10  # --- 분류 갯수
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2  # --- 검증용 데이터 (20%)

model = tf.keras.models.Sequential()  # --- 모델 : Sequential
model.add(
    keras.layers.Dense(
        NB_CLASSES,  # --- 출력 갯수
        input_shape=(RESHAPED,),  # --- 입력 갯수
        name="dense_layer",
        activation="softmax",
    )
)  # --- Activation Function : softmax  <  Sigmoid

model.summary()
model.compile(
    optimizer="SGD",  # --- Optimizer : SGD
    loss="categorical_crossentropy",  # --- Loss Function : categorical_crossentropy
    metrics=["accuracy"],
)  # --- Metrics

# --- 학습
model.fit(
    X_train,
    Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT,
)

# --- 평가
test_loss, test_acc = model.evaluate(X_test, Y_test)
print("Test accuracy:", test_acc)

# --- 예측
predictions = model.predict(X_test)
