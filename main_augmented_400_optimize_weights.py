from model_2FCN_3units import *
# from data import *
from color_generator import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import gc
import matplotlib.pyplot as plt
import pickle
import math

load_model = tf.keras.models.load_model
EarlyStopping = tf.keras.callbacks.EarlyStopping


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

with open('/home/yxl1214/mets_segment/shuffled_train_valid_data_Otsu_multiplier_400.p3', 'rb') as f:
    X_train, X_val, y_train, y_val = pickle.load(f)

# one-hot encoding is not needed if model last layer output one output, not softmax with two output
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=2)
# y_train = np.column_stack((y_train, ~y_train))
# y_val = np.column_stack((y_val, ~y_val))

ntrain = len(X_train)
nval = len(X_val)

import tensorflow as tf

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

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
y_gen = [train_generator.__getitem__(0)[1] for i in range(steps_per_epoch)]
print(np.unique(y_gen, return_counts=True))

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
#         multi_slice_viewer(np.squeeze(X_batch[i, :, :, :, 0]))
#         multi_slice_viewer(np.squeeze(X_batch[i, :, :, :, 1]))
#         plt.show()
#         input("Press Enter to continue...")

# ---------------------- augmented fit_generator-------------------------
# model = load_model('/home/yxl1214/mets_segment/final_model_augment_weight20_LRe-5.h5')
all_weights = [10, 15, 20]
all_AUCs = np.zeros(len(all_weights))
i = 0
for weight in all_weights:
    if weight == 20:
        class_weight = {0: 1, 1: weight}
        model = CNN_color(input_size=(100, 100, 12, 2))

        ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
        model_checkpoint = ModelCheckpoint('CNN_augment_weight20_LRe-5_size400.hdf5', monitor='val_acc',
                                           verbose=1, save_best_only=True)

        history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=100,
                                      validation_data=val_generator,
                                      class_weight=class_weight, validation_steps=steps_per_epoch_val,
                                      callbacks=[model_checkpoint, earlystop])
        # save history
        with open('trainHistoryDict_augment_weight' + str(weight) + '_size400.p3', 'wb') as file_pi:
            pickle.dump(history.history, file_pi, protocol=4)

        # save model
        model.save('final_model_augment_weight' + str(weight) + '_LRe-5_Otsu_multiplier_size400.h5')
    else:
        model = load_model('final_model_augment_weight' + str(weight) + '_LRe-5_Otsu_multiplier_size400.h5')
        
    # calculate AUC
    y_predict_400 = model.predict(X_val)
    all_AUCs[i] = roc_auc_score(y_val, y_predict_400, average="macro")
    i = i + 1
ind = np.argmax(all_AUCs)
print('best weight index = ' + str(ind))
print(str(all_AUCs))

# -----------------   plot accuracy and loss------------------------------
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# # accuracy
# plt.plot(epochs, acc, 'b', label='Training accuracy')
# plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
# plt.legend()
# plt.figure()
#
# # loss
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.legend()
# plt.show()

# predict the validation set

# ---------------------- perform test after optimizing the model -------------------------------
