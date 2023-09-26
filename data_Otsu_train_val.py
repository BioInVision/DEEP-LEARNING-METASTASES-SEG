import tensorflow as tf
import numpy as np
import os
from os import listdir
import pickle
import SimpleITK

import glob
import skimage.io as io
import skimage.transform as trans
# from nibabel import load as load_nii
import nibabel as nib

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


class load_Data:

    def __init__(self,
                 positive_train_path1="D:\\Users\yxl1214\GFP mets segment\deep learning test MIP 400x400x50 multiplier\cancer positive for train",
                 negative_train_path1="D:\\Users\yxl1214\GFP mets segment\deep learning test MIP 400x400x50 multiplier\cancer negative for train",
                 positive_train_path2="D:\\Users\yxl1214\GFP mets segment\deep learning test MIP 400x400x50 multiplier\cancer positive for train",
                 negative_train_path2="D:\\Users\yxl1214\GFP mets segment\deep learning test MIP 400x400x50 multiplier\cancer negative for train",
                 positive_val_path="D:\\Users\yxl1214\GFP mets segment\deep learning test MIP 400x400x50 multiplier\cancer positive for validation",
                 negative_val_path="D:\\Users\yxl1214\GFP mets segment\deep learning test MIP 400x400x50 multiplier\cancer negative for validation",
                 positive_test_path = "D:\\Users\yxl1214\GFP mets segment\deep learning test MIP 400x400x50 multiplier\cancer positive for test",
                 negative_test_path = "D:\\Users\yxl1214\GFP mets segment\deep learning test MIP 400x400x50 multiplier\cancer negative for test",
                 depth=12):
        self.positive_train_path1 = positive_train_path1
        self.negative_train_path1 = negative_train_path1
        self.positive_train_path2 = positive_train_path2
        self.negative_train_path2 = negative_train_path2
        self.positive_val_path = positive_val_path
        self.negative_val_path = negative_val_path
        self.positive_test_path = positive_test_path
        self.negative_test_path = negative_test_path
        self.z = depth

    def iterate_folder(self, folder):
        for filename in sorted(listdir(folder)):
            if filename.endswith('.nii'):
                yield filename

    def unison_shuffled_copies(self, x, y):
        print('x length = ' + str(x.shape[0]))
        print('y length = ' + str(y.shape[0]))
        assert x.shape[0] == y.shape[0]
        p = np.random.permutation(y.shape[0])
        return x[p, :, :, :, :], y[p]

    def load_files(self, X, path, filenames, iter):
        for f in filenames:
            temp = nib.load(os.path.join(path, f))
            if len(X.shape) == 4:
                X[:, :, :, iter] = temp.get_fdata()
            elif len(X.shape) == 5:
                X[:, :, :, :, iter] = temp.get_fdata()
            else:
                raise ValueError('X dimension wrong')
            iter += 1

        return X, iter

    def create_unshuffled_data(self, X1, X0, positive_path, positive_filenames, negative_path, negative_filenames):
        iter = 0
        X1, iter = self.load_files(X1, positive_path, positive_filenames, iter)

        y1 = np.array([1] * len(positive_filenames))

        iter = 0
        X0, iter = self.load_files(X0, negative_path, negative_filenames, iter)

        y0 = np.array([0] * len(negative_filenames))

        # # load GFP channel only
        # X1 = X1[:, :, :, 0, :]
        # X0 = X0[:, :, :, 0, :]

        if len(X1.shape) == 4:
            X = np.concatenate((X1, X0), axis=3)
            X = X.transpose(3, 0, 1, 2)
            X = X.reshape(X.shape + (1,)).astype(np.float32)
        elif len(X1.shape) == 5:
            X = np.concatenate((X1, X0), axis=4)
            X = X.transpose(4, 0, 1, 2, 3)
            X = X.astype(np.float32)
        else:
            raise ValueError('X dimension wrong')

        y = np.concatenate((y1, y0), axis=0)

        return X, y

    def create_unshuffled_data_separate(self, X1, X0, positive_path1, positive_filenames1, positive_path2,
                                        positive_filenames2, negative_path1, negative_filenames1, negative_path2,
                                        negative_filenames2):
        iter = 0
        X1, iter = self.load_files(X1, positive_path1, positive_filenames1, iter)
        X1, iter = self.load_files(X1, positive_path2, positive_filenames2, iter)

        y1 = np.array([1] * (len(positive_filenames1) + len(positive_filenames2)))

        iter = 0
        X0, iter = self.load_files(X0, negative_path1, negative_filenames1, iter)
        X0, iter = self.load_files(X0, negative_path2, negative_filenames2, iter)

        y0 = np.array([0] * (len(negative_filenames1) + len(negative_filenames2)))

        # # load GFP channel only
        # X1 = X1[:, :, :, 0, :]
        # X0 = X0[:, :, :, 0, :]

        if len(X1.shape) == 4:
            X = np.concatenate((X1, X0), axis=3)
            X = X.transpose(3, 0, 1, 2)
            X = X.reshape(X.shape + (1,)).astype(np.float32)
        elif len(X1.shape) == 5:
            X = np.concatenate((X1, X0), axis=4)
            X = X.transpose(4, 0, 1, 2, 3)
            X = X.astype(np.float32)
        else:
            raise ValueError('X dimension wrong')

        y = np.concatenate((y1, y0), axis=0)

        return X, y


    def load_control_data(self):
        negative_filenames = list(self.iterate_folder(self.negative_train_path1))
        X = np.zeros((100, 100, self.z, 2, len(negative_filenames)))

        iter = 0
        X, iter = self.load_files(X, self.negative_train_path1, negative_filenames, iter)

        y = np.array([0] * len(negative_filenames))

        if len(X1.shape) == 4:
            X = X.transpose(3, 0, 1, 2)
            X = X.reshape(X.shape + (1,)).astype(np.float32)
        elif len(X1.shape) == 5:
            X = X.transpose(4, 0, 1, 2, 3)
            X = X.astype(np.float32)
        else:
            raise ValueError('X dimension wrong')
        
        return X, y

    def load_test_data(self):
        positive_filenames = list(self.iterate_folder(self.positive_test_path))
        negative_filenames = list(self.iterate_folder(self.negative_test_path))

        X1 = np.zeros((100, 100, self.z, 2, len(positive_filenames)))
        X0 = np.zeros((100, 100, self.z, 2, len(negative_filenames)))

        X, y = self.create_unshuffled_data(X1, X0, self.positive_test_path, positive_filenames,
                                           self.negative_test_path, negative_filenames)
        return X, y

    def load_train_val_data(self):
        # use mouse C4M1 and C4M3 as train, and C6M2 as validation
        train_positive_filenames = list(self.iterate_folder(self.positive_train_path))
        train_negative_filenames = list(self.iterate_folder(self.negative_train_path))

        X1_train = np.zeros((100, 100, self.z, len(train_positive_filenames)), dtype=np.float32)
        X0_train = np.zeros((100, 100, self.z, len(train_negative_filenames)), dtype=np.float32)

        X_train, y_train = self.create_unshuffled_data(X1_train, X0_train, self.positive_train_path,
                                                       train_positive_filenames,
                                                       self.negative_train_path, train_negative_filenames)

        val_positive_filenames = list(self.iterate_folder(self.positive_val_path))
        val_negative_filenames = list(self.iterate_folder(self.negative_val_path))

        X1_val = np.zeros((100, 100, self.z, len(val_positive_filenames)), dtype=np.float32)
        X0_val = np.zeros((100, 100, self.z, len(val_negative_filenames)), dtype=np.float32)

        X_val, y_val = self.create_unshuffled_data(X1_val, X0_val, self.positive_val_path, val_positive_filenames,
                                                   self.negative_val_path, val_negative_filenames)

        return X_train, X_val, y_train, y_val

    def load_train_val_data_seperate(self):
        # use mouse C4M1 and C4M3 as train, and C6M2 as validation
        positive_train_path1 = self.positive_train_path1
        positive_train_path2 = self.positive_train_path2

        negative_train_path1 = self.negative_train_path1
        negative_train_path2 = self.negative_train_path2

        train_positive_filenames1 = list(self.iterate_folder(positive_train_path1))
        train_negative_filenames1 = list(self.iterate_folder(negative_train_path1))

        train_positive_filenames2 = list(self.iterate_folder(positive_train_path2))
        train_negative_filenames2 = list(self.iterate_folder(negative_train_path2))

        val_positive_filenames = list(self.iterate_folder(self.positive_val_path))
        val_negative_filenames = list(self.iterate_folder(self.negative_val_path))

        
        X1_train = np.zeros((100, 100, self.z, 2, len(train_positive_filenames1) + len(train_positive_filenames2)),
                            dtype=np.float32)
        X0_train = np.zeros((100, 100, self.z, 2, len(train_negative_filenames1) + len(train_negative_filenames2)),
                            dtype=np.float32)

        X1_val = np.zeros((100, 100, self.z, 2, len(val_positive_filenames)), dtype=np.float32)
        X0_val = np.zeros((100, 100, self.z, 2, len(val_negative_filenames)), dtype=np.float32)

        X_train, y_train = self.create_unshuffled_data_separate(X1_train, X0_train, positive_train_path1,
                                                                train_positive_filenames1, positive_train_path2,
                                                                train_positive_filenames2,
                                                                negative_train_path1, train_negative_filenames1,
                                                                negative_train_path2, train_negative_filenames2)

        X_val, y_val = self.create_unshuffled_data(X1_val, X0_val, self.positive_val_path, val_positive_filenames,
                                                   self.negative_val_path, val_negative_filenames)

        return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    # load multi-level train and validation data
    level400 = load_Data(
        positive_train_path1="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input400\cancer positive for train2",
        negative_train_path1="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input400\cancer negative for train2",
        positive_train_path2="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input400\cancer positive for test",
        negative_train_path2="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input400\cancer negative for test",
        positive_val_path="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input400\cancer positive for validation",
        negative_val_path="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input400\cancer negative for validation",
        depth=12)

    X_train, X_val, y_train, y_val = level400.load_train_val_data_seperate()
    with open('C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\C6M3_C4M1train_C4M3test\unshuffled_train_valid_data_Otsu_multiplier_400.p3',
              'wb') as f:
        pickle.dump([X_train, X_val, y_train, y_val], f, protocol=4)

    level200 = load_Data(
        positive_train_path1="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input200\cancer positive for train2",
        negative_train_path1="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input200\cancer negative for train2",
        positive_train_path2="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input200\cancer positive for test",
        negative_train_path2="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input200\cancer negative for test",
        positive_val_path="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input200\cancer positive for validation",
        negative_val_path="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input200\cancer negative for validation",
        depth=12)

    X_train, X_val, y_train, y_val = level200.load_train_val_data_seperate()
    with open('C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\C6M3_C4M1train_C4M3test\unshuffled_train_valid_data_Otsu_multiplier_200.p3',
              'wb') as f:
        pickle.dump([X_train, X_val, y_train, y_val], f, protocol=4)

    level100 = load_Data(
        positive_train_path1="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input100\cancer positive for train2",
        negative_train_path1="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input100\cancer negative for train2",
        positive_train_path2="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input100\cancer positive for test",
        negative_train_path2="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input100\cancer negative for test",
        positive_val_path="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input100\cancer positive for validation",
        negative_val_path="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input100\cancer negative for validation",
        depth=12)

    X_train, X_val, y_train, y_val = level100.load_train_val_data_seperate()
    with open('C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\C6M3_C4M1train_C4M3test\unshuffled_train_valid_data_Otsu_multiplier_100.p3',
              'wb') as f:
        pickle.dump([X_train, X_val, y_train, y_val], f, protocol=4)

    # load multi-level test data
    level400_test = load_Data(
        positive_test_path="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input400\cancer positive for train1",
        negative_test_path="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input400\cancer negative for train1", depth=12)
    X, y = level400_test.load_test_data()
    with open('C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\C6M3_C4M1train_C4M3test\unshuffled_test_Otsu_multiplier_400.p3', 'wb') as f:
        pickle.dump([X, y], f, protocol=4)

    level200_test = load_Data(
        positive_test_path="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input200\cancer positive for train1",
        negative_test_path="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input200\cancer negative for train1", depth=12)
    X, y = level200_test.load_test_data()
    with open('C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\C6M3_C4M1train_C4M3test\unshuffled_test_Otsu_multiplier_200.p3', 'wb') as f:
        pickle.dump([X, y], f, protocol=4)

    level100_test = load_Data(
        positive_test_path="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input100\cancer positive for train1",
        negative_test_path="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input100\cancer negative for train1", depth=12)
    X, y = level100_test.load_test_data()
    with open('C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\C6M3_C4M1train_C4M3test\unshuffled_test_Otsu_multiplier_100.p3', 'wb') as f:
        pickle.dump([X, y], f, protocol=4)

    # load multi-level control mouse data, positive train path doesn't matter
    level400_control = load_Data(
        positive_train_path1="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input400\cancer positive for test",
        negative_train_path1="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input400\cancer negative for control", depth=12)
    X, y = level400_control.load_control_data()
    with open('C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\unshuffled_control_Otsu_multiplier_400.p3', 'wb') as f:
        pickle.dump([X, y], f, protocol=4)

    level200_control = load_Data(
        positive_train_path1="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input200\cancer positive for test",
        negative_train_path1="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input200\cancer negative for control",
        depth=12)
    X, y = level200_control.load_control_data()
    with open('C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\unshuffled_control_Otsu_multiplier_200.p3', 'wb') as f:
        pickle.dump([X, y], f, protocol=4)

    level100_control = load_Data(
        positive_train_path1="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input100\cancer positive for test",
        negative_train_path1="C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\Input100\cancer negative for control",
        depth=12)
    X, y = level100_control.load_control_data()
    with open('C:\\Users\yxl1214\Dropbox\C6M3 GFP\metastases_segmentation\deep_learning_data\unshuffled_control_Otsu_multiplier_100.p3', 'wb') as f:
        pickle.dump([X, y], f, protocol=4)

