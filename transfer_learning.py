from model_2FCN_3units import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

load_model = tf.keras.models.load_model
EarlyStopping = tf.keras.callbacks.EarlyStopping
Model = tf.keras.Model

def extractGFP(X):
    X_GFP = X[:, :, :, :, 0]
    X_GFP = X_GFP.reshape(X_GFP.shape + (1,))
    return X_GFP

def create_all_features(model100_path, model200_path, model400_path):
    # load training and validation data
    with open(
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C4M1train_C6M3test/unshuffled_train_valid_data_Otsu_multiplier_100.p3',
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C6M3_C4M1train_C4M3test/unshuffled_train_valid_data_Otsu_multiplier_100.p3',
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C4M1test/unshuffled_train_valid_data_Otsu_multiplier_100.p3',
            'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C6M2test/unshuffled_train_valid_data_Otsu_multiplier_100.p3',
            'rb') as f:
        X_train_100, X_val_100, y_train_100, y_val_100 = pickle.load(f)

    with open(
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C4M1train_C6M3test/unshuffled_train_valid_data_Otsu_multiplier_200.p3',
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C6M3_C4M1train_C4M3test/unshuffled_train_valid_data_Otsu_multiplier_200.p3',
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C4M1test/unshuffled_train_valid_data_Otsu_multiplier_200.p3',
            'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C6M2test/unshuffled_train_valid_data_Otsu_multiplier_200.p3',
            'rb') as f:
        X_train_200, X_val_200, y_train_200, y_val_200 = pickle.load(f)

    with open(
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C4M1train_C6M3test/unshuffled_train_valid_data_Otsu_multiplier_400.p3',
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C6M3_C4M1train_C4M3test/unshuffled_train_valid_data_Otsu_multiplier_400.p3',
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C4M1test/unshuffled_train_valid_data_Otsu_multiplier_400.p3',
            'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C6M2test/unshuffled_train_valid_data_Otsu_multiplier_400.p3',
            'rb') as f:
        X_train_400, X_val_400, y_train_400, y_val_400 = pickle.load(f)

    # load test data
    with open(
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C4M1train_C6M3test/unshuffled_test_Otsu_multiplier_100.p3',
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C6M3_C4M1train_C4M3test/unshuffled_test_Otsu_multiplier_100.p3',
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C4M1test/unshuffled_test_Otsu_multiplier_100.p3',
            'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C6M2test/unshuffled_test_Otsu_multiplier_100.p3',
            'rb') as f:
        X_test_100, y_test_100 = pickle.load(f)

    with open(
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C4M1train_C6M3test/unshuffled_test_Otsu_multiplier_200.p3',
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C6M3_C4M1train_C4M3test/unshuffled_test_Otsu_multiplier_200.p3',
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C4M1test/unshuffled_test_Otsu_multiplier_200.p3',
            'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C6M2test/unshuffled_test_Otsu_multiplier_200.p3',
            'rb') as f:
        X_test_200, y_test_200 = pickle.load(f)

    with open(
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C4M1train_C6M3test/unshuffled_test_Otsu_multiplier_400.p3',
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C6M3_C4M1train_C4M3test/unshuffled_test_Otsu_multiplier_400.p3',
            # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C4M1test/unshuffled_test_Otsu_multiplier_400.p3',
            'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C6M2test/unshuffled_test_Otsu_multiplier_400.p3',
            'rb') as f:
        X_test_400, y_test_400 = pickle.load(f)

    # load models
    DL_model_100 = load_model(model100_path)
    DL_model_200 = load_model(model200_path)
    DL_model_400 = load_model(model400_path)

    # load hand-crafted features
    X_train_features_ML = loadmat(
        # 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/deep_learning_data/C4M3_C4M1_feature_array_sorted_DL_Otsu_multiply.mat', squeeze_me=True)
        # 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/deep_learning_data/C6M3_C4M1_feature_array_sorted_DL_Otsu_multiply.mat', squeeze_me=True)
        # 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/deep_learning_data/C4M3_C6M3_feature_array_sorted_DL_Otsu_multiply.mat', squeeze_me=True)
        'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/deep_learning_data/C4M3_C6M3_feature_array_sorted_DL_Otsu_multiply.mat', squeeze_me=True)

    X_val_features_ML = loadmat('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/deep_learning_data/C4M1_feature_array_sorted_DL_Otsu_multiply.mat',
                                squeeze_me=True)

    X_test_features_ML = loadmat('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/deep_learning_data/C6M2_feature_array_sorted_DL_Otsu_multiply.mat',
                                 squeeze_me=True)

    # re-scale the inputs
    X_train_100 *= 1 / 255
    X_train_200 *= 1 / 255
    X_train_400 *= 1 / 255

    X_val_100 *= 1 / 255
    X_val_200 *= 1 / 255
    X_val_400 *= 1 / 255

    X_test_100 *= 1 / 255
    X_test_200 *= 1 / 255
    X_test_400 *= 1 / 255

    X_train_features_ML = X_train_features_ML['T_train_sort_array']
    X_val_features_ML = X_val_features_ML['T_C4M1_sort_array']
    X_test_features_ML = X_test_features_ML['T_C6M2_sort_array']

    # create one stage CNN features for three networks
    DL_model_100_2 = Model(DL_model_100.input, DL_model_100.layers[-2].output)
    # DL_model_100_2.summary()
    DL_model_200_2 = Model(DL_model_200.input, DL_model_200.layers[-2].output)
    DL_model_400_2 = Model(DL_model_400.input, DL_model_400.layers[-2].output)

    X_train_features_CNN_100 = create_CNN_features(X_train_100, DL_model_100_2)
    X_train_features_CNN_200 = create_CNN_features(X_train_200, DL_model_200_2)
    X_train_features_CNN_400 = create_CNN_features(X_train_400, DL_model_400_2)

    X_val_features_CNN_100 = create_CNN_features(X_val_100, DL_model_100_2)
    X_val_features_CNN_200 = create_CNN_features(X_val_200, DL_model_200_2)
    X_val_features_CNN_400 = create_CNN_features(X_val_400, DL_model_400_2)

    X_test_features_CNN_100 = create_CNN_features(X_test_100, DL_model_100_2)
    X_test_features_CNN_200 = create_CNN_features(X_test_200, DL_model_200_2)
    X_test_features_CNN_400 = create_CNN_features(X_test_400, DL_model_400_2)

    # stack CNN features and ML features

    X_train_features = np.concatenate((X_train_features_CNN_100, X_train_features_CNN_200, X_train_features_CNN_400, X_train_features_ML), axis=1)
    X_val_features = np.concatenate((X_val_features_CNN_100, X_val_features_CNN_200, X_val_features_CNN_400, X_val_features_ML), axis=1)
    X_test_features = np.concatenate((X_test_features_CNN_100, X_test_features_CNN_200, X_test_features_CNN_400, X_test_features_ML), axis=1)

    with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/DL_ML_data_new_label_Otsu_BF_oneStage_crossVal4_weight20.p3', 'wb') as f:
        pickle.dump([X_train_features, y_train_100, X_val_features, y_val_100, X_test_features, y_test_100], f, protocol=4)

def create_CNN_features(x, pre_model):
    features = pre_model.predict(x, batch_size=8)
    features_flatten = features.reshape((features.shape[0], 256))
    return features_flatten

def create_twoStage_CNN_features(x, model1, model2_path):
    model2 = load_model(model2_path)

    FC_model1 = Model(model1.input, model1.layers[-2].output)
    FC_model2 = Model(model2.input, model2.layers[-2].output)

    X_GFP = x[:, :, :, :, 0]
    X_GFP = X_GFP.reshape(X_GFP.shape + (1,))

    y_predict1 = model1.predict(X_GFP)

    hard_val_idx = y_predict1[:, 0] > 0.5
    x_hard = x[hard_val_idx, :, :, :, :]

    features1 = FC_model1.predict(X_GFP, batch_size=8)
    features_flatten = features1.reshape((features1.shape[0], 256))
    features2 = FC_model2.predict(x_hard, batch_size=8)
    features_flatten[hard_val_idx, :] = features2.reshape((features2.shape[0], 256))

    return features_flatten

class transfer_learning:
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.X_train_features, self.y_train, self.X_val_features, self.y_val, self.X_test_features, self.y_test = pickle.load(f)
        print(self.X_train_features[:, 0:-1].shape)

    def Performance(self, Model, Y, X):
        # Perforamnce of the model
        y_predict = Model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(Y, y_predict)
        # with open('decision_tree_ROC_GF.p3', 'wb') as f:
        #     pickle.dump([fpr, tpr], f, protocol=4)

        AUC = auc(fpr, tpr)
        print('the AUC is : %0.4f' % AUC)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % AUC)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

        precision, recall, _ = precision_recall_curve(Y, y_predict)

        APS = average_precision_score(Y, y_predict)
        print('the APS is : %0.4f' % APS)
        plt.figure()
        plt.plot(recall, precision, label='precision-recall curve (score = %0.4f)' % APS)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.show()

        thresholds = np.linspace(0.0, 1.0, 200)

        for t in thresholds:
            y_predict_binary = y_predict > t

            tn, fp, fn, tp = confusion_matrix(Y, y_predict_binary).ravel()
            print('threshold=' + str(t))
            print(tn, fp, fn, tp)

        # select the threshold desired and write out the classification result
        self.write_classification_toMat(y_predict, Y, 0.035)

    def write_classification_toMat(self, y_predict, groundtruth, threshold):
        FP_idx = []
        FN_idx = []
        TP_idx = []
        TN_idx = []

        y_predict_binary = y_predict > threshold
        print('y_predict size =' + str(len(y_predict)))
        print('groundtruth size =' + str(len(groundtruth)))

        for i in range(len(y_predict)):
            if groundtruth[i] == 1 and y_predict_binary[i] == 0:
                FN_idx.append(i + 1)
            if groundtruth[i] == 0 and y_predict_binary[i] == 1:
                FP_idx.append(i + 1)
            if groundtruth[i] == 0 and y_predict_binary[i] == 0:
                TN_idx.append(i + 1)
            if groundtruth[i] == 1 and y_predict_binary[i] == 1:
                TP_idx.append(i + 1)

        savemat('C6M2_prediction_idx' + str(threshold) + '.mat', {"TP_idx": TP_idx, "TN_idx": TN_idx,
                                                             "FP_idx": FP_idx, "FN_idx": FN_idx})


    def FP_FN_index(self, Model, Y, X, threshold):
        # find the FP and FN indices and output them into two txt files
        y_predict = Model.predict_proba(X)[:, 1]
        y_predict_binary = y_predict > threshold

        tn, fp, fn, tp = confusion_matrix(Y, y_predict_binary).ravel()
        print(tn, fp, fn, tp)

        FP_idx = []
        FN_idx = []

        for i in range(len(y_predict)):
            if Y[i] == 1 and y_predict_binary[i] == 0:
                FN_idx.append(i + 1)
            if Y[i] == 0 and y_predict_binary[i] == 1:
                FP_idx.append(i + 1)

        # write the FP and FN indices into .mat files
        savemat('FP_idx'+str(threshold)+'.mat', {"FP_idx": FP_idx})
        savemat('FN_idx'+str(threshold)+'.mat', {"FN_idx": FN_idx})

    def optimize_decision_tree(self, feature_set):
        # feature_set = 0: use all features
        # feature_set = 1: use only CNN features
        # feature_set = 2: use only ML features

        class_weights_dict = {}
        class_weights_list = []
        for i in range(10, 1000, 20):
            class_weights_dict[0] = 1
            class_weights_dict[1] = i
            class_weights_list.append(class_weights_dict.copy())

        param_space = {
            'max_depth': hp.choice('max_depth', range(1, 100)),
            'max_features': hp.choice('max_features', range(1, 500)),
            'min_samples_split': hp.choice('min_samples_split', range(5, 150)),
            'min_samples_leaf': hp.choice('min_samples_leaf', range(5, 150)),
            'class_weight': hp.choice('class_weight', [None, 'balanced', 'balanced_subsample'] + class_weights_list),
            'n_estimators': hp.choice('n_estimators', range(50, 1000)),
            'criterion': hp.choice('criterion', ["gini", "entropy"])}

        self.best = 0

        rstate = np.random.RandomState(42)
        trials = Trials()

        if feature_set == 0:
            self.best = fmin(self.f_all, param_space, algo=tpe.suggest, max_evals=100, trials=trials, rstate=rstate)
        elif feature_set == 1:
            self.best = fmin(self.f_CNN, param_space, algo=tpe.suggest, max_evals=100, trials=trials, rstate=rstate)
        elif feature_set == 2:
            self.best = fmin(self.f_ML, param_space, algo=tpe.suggest, max_evals=100, trials=trials, rstate=rstate)

        print('best:')
        print(self.best)

        best_params = space_eval(param_space, trials.argmin)

        return best_params

    def f_all(self, params):
        acc = self.acc_model_all_features(params)
        if acc > self.best:
            self.best = acc
        print('new best:', self.best, params)
        return {'loss': -acc, 'status': STATUS_OK}

    def f_CNN(self, params):
        acc = self.acc_model_CNN_features(params)
        if acc > self.best:
            self.best = acc
        print('new best:', self.best, params)
        return {'loss': -acc, 'status': STATUS_OK}

    def f_ML(self, params):
        acc = self.acc_model_ML_features(params)
        if acc > self.best:
            self.best = acc
        print('new best:', self.best, params)
        return {'loss': -acc, 'status': STATUS_OK}

    def acc_model_all_features(self, params):
        clf = RandomForestClassifier(**params, n_jobs=-1)
        clf.fit(self.X_train_features[:, 0:-1], self.y_train)
        y_predict = clf.predict_proba(self.X_val_features[:, 0:-1])[:, 1]

        # use AUC as the object to optimize
        fpr, tpr, _ = roc_curve(self.y_val, y_predict)
        AUC = auc(fpr, tpr)

        # use F1 score as the object to optimize
        # y_predict_binary = y_predict > 0.1
        # AUC = f1_score(self.y_val, y_predict_binary)
        return AUC

    def acc_model_CNN_features(self, params):
        clf = RandomForestClassifier(**params, n_jobs=-1)
        clf.fit(self.X_train_features[:, 0:-30], self.y_train)
        y_predict = clf.predict_proba(self.X_val_features[:, 0:-30])[:, 1]

        # use AUC as the object to optimize
        fpr, tpr, _ = roc_curve(self.y_val, y_predict)
        AUC = auc(fpr, tpr)

        # use F1 score as the object to optimize
        # y_predict_binary = y_predict > 0.1
        # AUC = f1_score(self.y_val, y_predict_binary)
        return AUC

    def acc_model_ML_features(self, params):
        clf = RandomForestClassifier(**params, n_jobs=-1)
        clf.fit(self.X_train_features[:, -30:-1], self.y_train)
        y_predict = clf.predict_proba(self.X_val_features[:, -30:-1])[:, 1]

        # use AUC as the object to optimize
        fpr, tpr, _ = roc_curve(self.y_val, y_predict)
        AUC = auc(fpr, tpr)

        # use F1 score as the object to optimize
        # y_predict_binary = y_predict > 0.1
        # AUC = f1_score(self.y_val, y_predict_binary)
        return AUC


if __name__ == "__main__":
    # create all features, 1: 100, 2:200, 3:400
    # model100_path = 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_network/C4M3_C4M1train_C6M2val_C6M3test/final_model_augment_weight20_LRe-5_Otsu_multiplier_size100.h5'
    # model200_path = 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_network/C4M3_C4M1train_C6M2val_C6M3test/final_model_augment_weight20_LRe-5_Otsu_multiplier_size200.h5'
    # model400_path = 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_network/C4M3_C4M1train_C6M2val_C6M3test/final_model_augment_weight20_LRe-5_Otsu_multiplier_size400.h5'

    # model100_path = 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_network/C4M1_C6M3train_C6M2val_C4M3test/final_model_augment_weight20_LRe-5_Otsu_multiplier_size100.h5'
    # model200_path = 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_network/C4M1_C6M3train_C6M2val_C4M3test/final_model_augment_weight20_LRe-5_Otsu_multiplier_size200.h5'
    # model400_path = 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_network/C4M1_C6M3train_C6M2val_C4M3test/final_model_augment_weight20_LRe-5_Otsu_multiplier_size400.h5'

    # model100_path = 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_network/C4M3_C6M3train_C6M2val_C4M1test/final_model_augment_weight10_LRe-5_Otsu_multiplier_size100.h5'
    # model200_path = 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_network/C4M3_C6M3train_C6M2val_C4M1test/final_model_augment_weight10_LRe-5_Otsu_multiplier_size200.h5'
    # model400_path = 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_network/C4M3_C6M3train_C6M2val_C4M1test/final_model_augment_weight10_LRe-5_Otsu_multiplier_size400.h5'

    model100_path = 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_network/C4M3_C6M3train_C4M1val_C6M2test/final_model_augment_weight20_LRe-5_Otsu_multiplier_size100.h5'
    model200_path = 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_network/C4M3_C6M3train_C4M1val_C6M2test/final_model_augment_weight20_LRe-5_Otsu_multiplier_size200.h5'
    model400_path = 'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_network/C4M3_C6M3train_C4M1val_C6M2test/final_model_augment_weight20_LRe-5_Otsu_multiplier_size400.h5'

    create_all_features(model100_path, model200_path, model400_path)

    # load created features and perform transfer learning
    # feature_set = 0: use all features
    # feature_set = 1: use only CNN features
    # feature_set = 2: use only ML features

    tf_learning = transfer_learning(data_path='C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/DL_ML_data_new_label_Otsu_BF_oneStage_crossVal4_weight20.p3')
    feature_set_idx = 0
    best_params = tf_learning.optimize_decision_tree(feature_set_idx)
    print(best_params)

    if feature_set_idx == 0:
        rnd_clf = RandomForestClassifier(**best_params, n_jobs=-1)
        rnd_clf.fit(tf_learning.X_train_features[:, 0:-1], tf_learning.y_train)

        # save the trained random forest classifier
        # with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/trained_rnd_clf_all_features_multiscale_BF_oneStage_crossVal4_weight20.p3', 'wb') as f:
        #     pickle.dump(rnd_clf, f, protocol=4)

        # save the trained random forest classifier
        # with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/trained_rnd_clf_all_features_multiscale_BF_oneStage_crossVal4_weight20.p3', 'rb') as f:
        #     rnd_clf = pickle.load(f)

        tf_learning.Performance(rnd_clf, tf_learning.y_val, tf_learning.X_val_features[:, 0:-1])
        tf_learning.Performance(rnd_clf, tf_learning.y_test, tf_learning.X_test_features[:, 0:-1])

    elif feature_set_idx == 1:
        rnd_clf = RandomForestClassifier(**best_params, n_jobs=-1)
        rnd_clf.fit(tf_learning.X_train_features[:,0:256], tf_learning.y_train)

        tf_learning.Performance(rnd_clf, tf_learning.y_val, tf_learning.X_val_features[:,0:256])
        tf_learning.Performance(rnd_clf, tf_learning.y, tf_learning.X_test_features[:,0:256])

    elif feature_set_idx == 2:
        rnd_clf = RandomForestClassifier(**best_params, n_jobs=-1)
        rnd_clf.fit(tf_learning.X_train_features[:, 256:285], tf_learning.y_train)

        with open('trained_rnd_clf_ML_features.p3', 'wb') as f:
            pickle.dump(rnd_clf, f, protocol=4)

        tf_learning.Performance(rnd_clf, tf_learning.y_val, tf_learning.X_val_features[:, 256:285])
        tf_learning.Performance(rnd_clf, tf_learning.y, tf_learning.X_test_features[:, 256:285])
