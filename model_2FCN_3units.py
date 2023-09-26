import numpy as np
import os
# import skimage.io as io
# import skimage.transform as trans
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

"the following is due to failure to resolve reference in Pycharm for the keras library"
Input = tf.keras.layers.Input
Conv3D = tf.keras.layers.Conv3D
BatchNormalization = tf.keras.layers.BatchNormalization
Activation = tf.keras.layers.Activation
MaxPooling3D = tf.keras.layers.MaxPooling3D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Model = tf.keras.Model
Adam = tf.keras.optimizers.Adam

def binary_focal_loss(gamma=2., alpha=.5):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed

def Tversky_F1_loss(alpha = 0.1):
    """
    F1 loss with Tversky weights for false positive and false negative
    alpha is the weight for False positive and 1-alpha is the weight for false negative
    """
    def Tversky_F1_loss_fixed(y_true, y_pred):
        TP = tf.where(tf.equal(y_true, 1), y_pred, tf.zeros_like(y_pred))
        FP = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        FN = tf.where(tf.equal(y_true, 1), 1 - y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()

        f1 = (K.sum(TP) + epsilon) / (K.sum(TP) + alpha * K.sum(FP) + (1-alpha) * K.sum(FN) + epsilon)
        return 1. - f1

    return Tversky_F1_loss_fixed
    

def CNN_new(pretrained_weights=None, input_size=(100, 100, 25, 2), learnRate=1e-5):
    inputs = Input(input_size)
    conv1_1 = Conv3D(32, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(inputs)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)

    conv1_2 = Conv3D(32, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = Activation('relu')(conv1_2)

    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1_2)

    conv2_1 = Conv3D(64, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(pool1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = Activation('relu')(conv2_1)

    conv2_2 = Conv3D(64, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = Activation('relu')(conv2_2)

    conv2_3 = Conv3D(64, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(conv2_2)
    conv2_3 = BatchNormalization()(conv2_3)
    conv2_3 = Activation('relu')(conv2_3)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2_3)

    conv3_1 = Conv3D(128, kernel_size=(3,3,3), activation=None, padding='same', kernel_initializer='he_normal')(pool2)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)

    conv3_2 = Conv3D(128, kernel_size=(3,3,3), activation=None, padding='same', kernel_initializer='he_normal')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)

    conv3_3 = Conv3D(128, kernel_size=(3,3,3), activation=None, padding='same', kernel_initializer='he_normal')(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_3 = Activation('relu')(conv3_3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3_3)
    
    conv4_1 = Conv3D(256, kernel_size=(3, 3, 1), activation=None, padding='same', kernel_initializer='he_normal')(pool3)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_1 = Activation('relu')(conv4_1)

    conv4_2 = Conv3D(256, kernel_size=(3, 3, 1), activation=None, padding='same', kernel_initializer='he_normal')(
        conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_2 = Activation('relu')(conv4_2)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4_2)

    flat3 = Flatten()(pool4)
    dense1 = Dense(256, activation='relu')(flat3)
    dense2 = Dense(2, activation='softmax')(dense1)

    model = Model(inputs=inputs, outputs=dense2)

    model.compile(optimizer=Adam(lr=learnRate), loss='categorical_crossentropy', metrics=["accuracy"])
    # model.compile(optimizer=Adam(lr=1e-6), loss=[binary_focal_loss(alpha=.5, gamma=2)], metrics=["accuracy"])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def CNN_color(pretrained_weights=None, input_size=(100, 100, 25, 2), learnRate=1e-5):
    inputs = Input(input_size)
    conv1_1 = Conv3D(32, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(inputs)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)

    conv1_2 = Conv3D(32, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = Activation('relu')(conv1_2)

    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1_2)

    conv2_1 = Conv3D(64, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(pool1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = Activation('relu')(conv2_1)

    conv2_2 = Conv3D(64, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = Activation('relu')(conv2_2)

    conv2_3 = Conv3D(64, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(conv2_2)
    conv2_3 = BatchNormalization()(conv2_3)
    conv2_3 = Activation('relu')(conv2_3)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2_3)

    conv3_1 = Conv3D(128, kernel_size=(3,3,3), activation=None, padding='same', kernel_initializer='he_normal')(pool2)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)

    conv3_2 = Conv3D(128, kernel_size=(3,3,3), activation=None, padding='same', kernel_initializer='he_normal')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)

    conv3_3 = Conv3D(128, kernel_size=(3,3,3), activation=None, padding='same', kernel_initializer='he_normal')(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_3 = Activation('relu')(conv3_3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3_3)

    flat3 = Flatten()(pool3)
    dense1 = Dense(256, activation='sigmoid')(flat3)
    dense2 = Dense(2, activation='softmax')(dense1)

    model = Model(inputs=inputs, outputs=dense2)

    model.compile(optimizer=Adam(lr=learnRate), loss='binary_crossentropy', metrics=["accuracy"])
    # model.compile(optimizer=Adam(lr=1e-6), loss=[binary_focal_loss(alpha=.5, gamma=2)], metrics=["accuracy"])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def CNN_color_sigmoid(pretrained_weights=None, input_size=(100, 100, 25, 2), learnRate=1e-5):
    inputs = Input(input_size)
    conv1_1 = Conv3D(32, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(inputs)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)

    conv1_2 = Conv3D(32, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = Activation('relu')(conv1_2)

    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1_2)

    conv2_1 = Conv3D(64, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(pool1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = Activation('relu')(conv2_1)

    conv2_2 = Conv3D(64, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = Activation('relu')(conv2_2)

    conv2_3 = Conv3D(64, kernel_size=(5,5,3), activation=None, padding='same', kernel_initializer='he_normal')(conv2_2)
    conv2_3 = BatchNormalization()(conv2_3)
    conv2_3 = Activation('relu')(conv2_3)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2_3)

    conv3_1 = Conv3D(128, kernel_size=(3,3,3), activation=None, padding='same', kernel_initializer='he_normal')(pool2)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)

    conv3_2 = Conv3D(128, kernel_size=(3,3,3), activation=None, padding='same', kernel_initializer='he_normal')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)

    conv3_3 = Conv3D(128, kernel_size=(3,3,3), activation=None, padding='same', kernel_initializer='he_normal')(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_3 = Activation('relu')(conv3_3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3_3)

    flat3 = Flatten()(pool3)
    dense1 = Dense(256, activation='sigmoid')(flat3)
    dense2 = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=inputs, outputs=dense2)

    model.compile(optimizer=Adam(lr=learnRate), loss='binary_crossentropy', metrics=["accuracy"])
    # model.compile(optimizer=Adam(lr=1e-6), loss=[binary_focal_loss(alpha=.5, gamma=2)], metrics=["accuracy"])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
