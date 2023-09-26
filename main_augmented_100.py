from model_2FCN_3units import *
# from data import *
from color_generator import *
# from generator import *
from sklearn.model_selection import train_test_split
import gc
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
load_weights = tf.keras.Model.load_weights
load_model = tf.keras.models.load_model
EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint


def unison_shuffled_copies(x, y):
    assert x.shape[0] == y.shape[0]
    p = np.random.permutation(y.shape[0])
    return x[p, :, :, :, :], y[p]

# 3D image display functions
def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[2] // 2
    ax.imshow(volume[:, :, ax.index], aspect='auto')
    fig.canvas.mpl_connect('key_press_event', process_key)


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
    ax.images[0].set_array(volume[:, :, ax.index])


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[2]
    ax.images[0].set_array(volume[:, :, ax.index])


# learning rate scheduler
def scheduler(epoch):
    if epoch < 1:
        return 0.0001
    else:
        print(0.0001 * math.exp(0.1 * (1. - float(epoch))))
        return 0.0001 * math.exp(0.1 * (1. - float(epoch)))


# early stopping
earlystop = EarlyStopping(monitor='val_acc',
                          min_delta=0,
                          patience=20,
                          verbose=1)

# with open('/home/yxl1214/mets_segment/shuffled_train_valid_data_Otsu_multiplier_100.p3', 'rb') as f:
#     X_train, X_val, y_train, y_val = pickle.load(f)
# 
# # GFP only
# X_train = X_train[:, :, :, :, 0]
# X_train = X_train.reshape(X_train.shape + (1,))
# 
# X_val = X_val[:, :, :, :, 0]
# X_val = X_val.reshape(X_val.shape + (1,))
# 
# with open('/home/yxl1214/mets_segment/shuffled_train_valid_GFP_Otsu_multiplier_100.p3', 'wb') as f:
#     pickle.dump([X_train, X_val, y_train, y_val], f, protocol=4)
# 
# print("GFP data created")

with open('/home/yxl1214/mets_segment/unshuffled_train_valid_data_Otsu_multiplier_100.p3', 'rb') as f:
    X_train, X_val, y_train, y_val = pickle.load(f)

# shuffle data
X_train, y_train = unison_shuffled_copies(X_train, y_train)
X_val, y_val = unison_shuffled_copies(X_val, y_val)

# one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=2)

ntrain = len(X_train)
nval = len(X_val)

# -------------------------------- With data augmentation-----------------------------------------------------
batch_size = 8
train_datagen = colorImageDataGenerator(rotation_range=90,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         zoom_range=0.1,
                                         rescale = 1/255,
                                         brightness_range=[0.8, 1.2],
                                         data_format='channels_last')
train_generator = BalancedDataGenerator(X_train, y_train, train_datagen, batch_size=batch_size)
steps_per_epoch = train_generator.steps_per_epoch

# check the number of cancer positive and negative candidate in one batch
# y_gen = [train_generator.__getitem__(0)[1] for i in range(steps_per_epoch)]
# print(np.unique(y_gen, return_counts=True))

val_datagen = colorImageDataGenerator(rotation_range=90,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       zoom_range=0.1,
                                       rescale = 1/255,
                                       brightness_range=[0.8, 1.2],
                                       data_format='channels_last')
val_generator = BalancedDataGenerator(X_val, y_val, val_datagen, batch_size=batch_size)

steps_per_epoch_val = val_generator.steps_per_epoch
# ------------- visualize the image in one batch-------------------------
# for X_batch, y_batch in train_generator:
#     for i in range(len(y_batch)):
#         multi_slice_viewer(np.squeeze(X_batch[i,:,:,:,:]))
#        input("Press Enter to continue...")

model = CNN_color(input_size=(100, 100, 12, 2))

model_checkpoint = ModelCheckpoint('/home/yxl1214/mets_segment/CNN_augment_weight20_LRe-5_size100.hdf5', monitor='val_acc',
                                   verbose=1, save_best_only=True)

# ---------------------- augmented fit_generator-------------------------
# model = load_model('/home/yxl1214/mets_segment/GFP_weight20_LRe-5_Otsu_multiplier_size100.h5')
class_weight = {0: 1, 1: 20}
history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=100,
                              validation_data=val_generator,
                              class_weight=class_weight, validation_steps=steps_per_epoch_val,
                              callbacks=[model_checkpoint, earlystop])
# save history
with open('/home/yxl1214/mets_segment/trainHistoryDict_augment_weight20_size100.p3', 'wb') as file_pi:
    pickle.dump(history.history, file_pi, protocol=4)

# save model
model.save('/home/yxl1214/mets_segment/softmax_layer3_augment_weight20_LR1e-5_batch8_size100_GFP.h5')

# -----------------   plot accuracy and loss------------------------------
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# accuracy
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.legend()
plt.figure()

# loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.legend()
plt.show()
