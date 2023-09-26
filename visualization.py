from model_2FCN_3units import *
import numpy as np
from matplotlib import pyplot as plt
# from tf_keras_vis import utils
# from tf_keras_vis.saliency import Saliency
# from tf_keras_vis.utils import normalize
# import vis
from keras_vis.vis.utils import utils as up
from keras_vis.vis.visualization import saliency
#from vis.utils import utils
#from vis.visualization import *
# from utils.visualization import *
load_model = tf.keras.models.load_model
Model = tf.keras.Model
K = tf.keras.backend
activations = tf.keras.activations
import pickle
import nibabel as nib
get_default_graph = tf.compat.v1.get_default_graph()

# tf.compat.v1.disable_eager_execution()
# print(tf.compat.v1.get_default_graph())

def model_modifier(m):
    m.layers[-1].activation = activations.linear
    return m


def loss(output):
    return (output[0][0], output[0][0])


# 3D image display functions
def multi_slice_viewer(volume, display_mode=1):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[2] // 2
    if display_mode == 0:
        # display grayscale image
        ax.imshow(volume[:, :, ax.index], aspect='equal', cmap='gray')
    else:
        # visualize color map
        ax.imshow(volume[:, :, ax.index], aspect='equal', cmap='jet')
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
    print("slice # =" + str(ax.index))


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[2]
    ax.images[0].set_array(volume[:, :, ax.index])
    print("slice # =" + str(ax.index))

# load images positive
img2 = nib.load("D:\YIQIAO DEEP LEARNING METASTASES SEG\DATA\combined_volume002151.nii") #TP big lung
# img2 = nib.load("D:\\Users\yxl1214\GFP mets segment\multilevel 100x100x16 new groundtruth wColor\Input100\cancer positive for train1\combined_volume001874.nii") #TP small lung
# img2 = nib.load("D:\\Users\yxl1214\GFP mets segment\multilevel 100x100x16 new groundtruth wColor\Input200\cancer positive for train1\combined_volume006267.nii") # TP liver

# load images negative
# img2 = nib.load("D:\\Users\yxl1214\GFP mets segment\multilevel 100x100x16 new groundtruth wColor\Input100\cancer negative for test\combined_volume001879.nii") #TN liver
# img2 = nib.load("D:\\Users\yxl1214\GFP mets segment\multilevel 100x100x16 new groundtruth wColor\Input400\cancer negative for train1\combined_volume005085.nii") #TN spine
#img2 = nib.load("D:\\Users\yxl1214\GFP mets segment\multilevel 100x100x16 new groundtruth wColor\Input400\cancer negative for train1\combined_volume002162.nii") #TN GI tract

# img2 = nib.load("D:\\Users\yxl1214\GFP mets segment\multilevel 400x400x48(100x100x12) new groundtruth wColor\Input400\cancer positive for train1\combined_volume002151.nii") #TP big
# img2 = nib.load("D:\\Users\yxl1214\GFP mets segment\multilevel 400x400x48(100x100x12) new groundtruth wColor\Input400\cancer positive for train1\combined_volume001874.nii") #TP small
# img2 = nib.load("D:\\Users\yxl1214\GFP mets segment\multilevel 400x400x48(100x100x12) new groundtruth wColor\Input400\cancer negative for test\combined_volume002788.nii") #FP
# img2 = nib.load("D:\\Users\yxl1214\GFP mets segment\multilevel 400x400x48(100x100x12) new groundtruth wColor\Input100\cancer negative for test\combined_volume001879.nii") #TN


# load negative
# img2 = nib.load("D:\\Users\yxl1214\GFP mets segment\multilevel 100x100x16 wColor\Input100\cancer negative for train1\combined_volume003270.nii")

img2 = img2.get_fdata()
# X = np.zeros((1, 100, 100, 12, 2))
X = np.zeros((1, 100, 100, 12, 2))
X[0] = img2[:, :, :12, :]

# X = np.zeros((100, 100, 16, 2, 2))
# X[:, :, :, :, 0] = img1.get_fdata()
# X[:, :, :, :, 1] = img2.get_fdata()
X *= 1 / 255
print(X.shape)
# model = load_model('D:/Users/yxl1214/GFP mets segment/saved network/100x100x16 4layers/small mets only/add_layer4/new_augment_weight20_LR1e-5_batch8_size100_smallMets.h5')
#model = load_model('D:/Users/yxl1214/GFP mets segment/saved network/100x100x16 4layers/big+small/softmax/softmax_layer4_augment_weight20_LR1e-5_batch8_size400.h5')
model = load_model('D:\YIQIAO DEEP LEARNING METASTASES SEG\saved_results\saved_network\C4M1_C6M3train_C6M2val_C4M3test\\final_model_augment_weight20_LRe-5_Otsu_multiplier_size200.h5')
model.summary()

y = model.predict(X, verbose=1)
print('prediction = ' + str(y))
# Create Saliency object using tf_keras_vis
# saliency = Saliency(model100, model_modifier, clone=False)
# saliency_map = saliency(loss, X)
# saliency_map = normalize(saliency_map)
# create saliency object using keras-vis
for layer in model.layers:
    print(layer.name)
dense_layer_idx = up.find_layer_idx(model, 'dense_7')
conv_layer_idx = up.find_layer_idx(model, 'conv3d_31')
model.layers[dense_layer_idx].activation = activations.linear
model = up.apply_modifications(model)
# model_visualization = ModelVisualization(model)
# saliency_map = visualize_saliency(model, dense_layer_idx, filter_indices=1, seed_input=X, backprop_modifier='guided')
#gradCAM_map = visualize_cam(model, dense_layer_idx, filter_indices=0, seed_input=X, penultimate_layer_idx=conv_layer_idx) # for negative
gradCAM_map = saliency.visualize_cam(model, dense_layer_idx, filter_indices=1, seed_input=X, penultimate_layer_idx=conv_layer_idx) # for positive

# multi_slice_viewer(saliency_map)
multi_slice_viewer(gradCAM_map)
multi_slice_viewer(img2[:, :, :, 1], display_mode=0)
multi_slice_viewer(img2[:, :, :, 0], display_mode=0)
plt.show()
# input("Press Enter to continue...")

# for X_batch, y_batch in train_generator:
#     for i in range(len(y_batch)):
#         multi_slice_viewer(np.squeeze(X_batch[i, :, :, :, 0]))
#         multi_slice_viewer(np.squeeze(X_batch[i, :, :, :, 1]))
#         plt.show()
#         input("Press Enter to continue...")

