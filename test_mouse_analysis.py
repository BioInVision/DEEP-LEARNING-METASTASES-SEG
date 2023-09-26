from model_2FCN_3units import *
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

load_model = tf.keras.models.load_model
EarlyStopping = tf.keras.callbacks.EarlyStopping
Model = tf.keras.Model


class FP_FN_analysis_multilevel:
    def __init__(self, model100_path, model200_path, model400_path):
        with open(
                # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C6M3_C4M1train_C4M3test/unshuffled_test_Otsu_multiplier_100.p3',
                'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C4M1train_C6M3test/unshuffled_test_Otsu_multiplier_100.p3',
                # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C4M1test/unshuffled_test_Otsu_multiplier_100.p3',
                # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C6M2test/unshuffled_test_Otsu_multiplier_100.p3',
                'rb') as f:
            self.X_test_100, self.y_test_100 = pickle.load(f)

        with open(
                # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C6M3_C4M1train_C4M3test/unshuffled_test_Otsu_multiplier_200.p3',
                'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C4M1train_C6M3test/unshuffled_test_Otsu_multiplier_200.p3',
                # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C4M1test/unshuffled_test_Otsu_multiplier_200.p3',
                # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C6M2test/unshuffled_test_Otsu_multiplier_200.p3',
                'rb') as f:
            self.X_test_200, self.y_test_200 = pickle.load(f)

        with open(
                # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C6M3_C4M1train_C4M3test/unshuffled_test_Otsu_multiplier_400.p3',
                'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C4M1train_C6M3test/unshuffled_test_Otsu_multiplier_400.p3',
                # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C4M1train_C6M3test/unshuffled_test_GFP_Otsu_multiplier_400.p3',
                # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C4M1test/unshuffled_test_Otsu_multiplier_400.p3',
                # 'D:/Users/yxl1214/GFP mets segment/multilevel 400x400x48(100x100x12) new groundtruth wColor/C4M3_C6M3train_C6M2test/unshuffled_test_Otsu_multiplier_400.p3',
                'rb') as f:
            self.X_test_400, self.y_test_400 = pickle.load(f)

        # re-scale the inputs
        self.X_test_100 *= 1 / 255
        self.X_test_200 *= 1 / 255
        self.X_test_400 *= 1 / 255

        X_test_features_ML = loadmat(
            'C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/deep_learning_data/C6M3_feature_array_sorted_DL_Otsu_multiply.mat',
            squeeze_me=True)
        self.X_test_features_ML = X_test_features_ML['T_C6M3_sort_array']
        self.X_test_volume = self.X_test_features_ML[:, 15]

        self.DL_model_100 = load_model(model100_path)
        self.DL_model_200 = load_model(model200_path)
        self.DL_model_400 = load_model(model400_path)

    def prediction_for_different_sizes(self, y_predict_100, y_predict_200, y_predict_400, idx100, idx200, idx400):
        predict_100_small = y_predict_100[idx100]
        predict_100_med = y_predict_100[idx200]
        predict_100_large = y_predict_100[idx400]

        predict_200_small = y_predict_200[idx100]
        predict_200_med = y_predict_200[idx200]
        predict_200_large = y_predict_200[idx400]

        predict_400_small = y_predict_400[idx100]
        predict_400_med = y_predict_400[idx200]
        predict_400_large = y_predict_400[idx400]

        print('level 100 small prediction mean = ' + str(np.mean(predict_100_small)))
        print('level 100 med prediction mean = ' + str(np.mean(predict_100_med)))
        print('level 100 large prediction mean = ' + str(np.mean(predict_100_large)))

        print('level 200 small prediction mean = ' + str(np.mean(predict_200_small)))
        print('level 200 med prediction mean = ' + str(np.mean(predict_200_med)))
        print('level 200 large prediction mean = ' + str(np.mean(predict_200_large)))

        print('level 400 small prediction mean = ' + str(np.mean(predict_400_small)))
        print('level 400 med prediction mean = ' + str(np.mean(predict_400_med)))
        print('level 400 large prediction mean = ' + str(np.mean(predict_400_large)))

    def write_classification_toMat(self, y_predict, threshold):
        FP_idx = []
        FN_idx = []
        TP_idx = []
        TN_idx = []

        y_predict_binary = y_predict > threshold
        for i in range(len(y_predict)):
            if self.y_test_100[i] == 1 and y_predict_binary[i] == 0:
                FN_idx.append(i + 1)
            if self.y_test_100[i] == 0 and y_predict_binary[i] == 1:
                FP_idx.append(i + 1)
            if self.y_test_100[i] == 0 and y_predict_binary[i] == 0:
                TN_idx.append(i + 1)
            if self.y_test_100[i] == 1 and y_predict_binary[i] == 1:
                TP_idx.append(i + 1)

        savemat('C6M3_prediction_idx' + str(threshold) + '.mat', {"TP_idx": TP_idx, "TN_idx": TN_idx,
                                                             "FP_idx": FP_idx, "FN_idx": FN_idx})


    def FP_FN_index_DL(self):
        y_predict_100 = self.DL_model_100.predict(self.X_test_100)
        y_predict_200 = self.DL_model_200.predict(self.X_test_200)
        y_predict_400 = self.DL_model_400.predict(self.X_test_400)

        if y_predict_100.shape[1] == 2:
            # softmax prediction
            y_predict_100 = y_predict_100[:, 1]
            y_predict_200 = y_predict_200[:, 1]
            y_predict_400 = y_predict_400[:, 1]
        else:
            # sigmoid prediction
            y_predict_100 = y_predict_100[:, 0]
            y_predict_200 = y_predict_200[:, 0]
            y_predict_400 = y_predict_400[:, 0]

        # compute the probability for three size ranges from each network
        # size small: <=50x50x6
        # size med: 50x50x6 - 100x100x12
        # size large: >100x100x12
        # self.prediction_for_different_sizes(y_predict_100, y_predict_200, y_predict_400, idx100, idx200, idx400)

        y_predict = 0.3 * y_predict_100 + 0.4 * y_predict_200 + 0.3 * y_predict_400

        # calculate ROC for each level and combined level
        fpr, tpr, thresholds = roc_curve(self.y_test_100, y_predict_100)
        print('level 100: ' + str(auc(fpr, tpr)))
        # with open('scale100_ROC_GF.p3', 'wb') as f:
        #     pickle.dump([fpr, tpr], f, protocol=4)

        fpr, tpr, thresholds = roc_curve(self.y_test_100, y_predict_200)
        print('level 200: ' + str(auc(fpr, tpr)))
        # with open('scale200_ROC_GF.p3', 'wb') as f:
        #     pickle.dump([fpr, tpr], f, protocol=4)

        fpr, tpr, thresholds = roc_curve(self.y_test_100, y_predict_400)
        print('level 400: ' + str(auc(fpr, tpr)))
        # with open('scale400_ROC_GF.p3', 'wb') as f:
        #     pickle.dump([fpr, tpr], f, protocol=4)

        fpr, tpr, thresholds = roc_curve(self.y_test_100, y_predict)
        print('combined level: ' + str(auc(fpr, tpr)))
        # with open('multiscale_ROC_GF.p3', 'wb') as f:
        #     pickle.dump([fpr, tpr], f, protocol=4)

        # calculate precision-recall AUC
        precision, recall, _ = precision_recall_curve(self.y_test_100, y_predict_100)
        print('level 100 PR: ' + str(auc(recall, precision)))
        # with open('scale100_PR_GF.p3', 'wb') as f:
        #     pickle.dump([precision, recall], f, protocol=4)

        precision, recall, _ = precision_recall_curve(self.y_test_100, y_predict_200)
        print('level 200 PR: ' + str(auc(recall, precision)))
        # with open('scale200_PR_GF.p3', 'wb') as f:
        #     pickle.dump([precision, recall], f, protocol=4)

        precision, recall, _ = precision_recall_curve(self.y_test_100, y_predict_400)
        print('level 400 PR: ' + str(auc(recall, precision)))
        # with open('scale400_PR_GF.p3', 'wb') as f:
        #     pickle.dump([precision, recall], f, protocol=4)

        precision, recall, _ = precision_recall_curve(self.y_test_100, y_predict)
        print('combined level PR: ' + str(auc(recall, precision)))
        # with open('multiscale_PR_GF.p3', 'wb') as f:
        #     pickle.dump([precision, recall], f, protocol=4)

        self.plot_all_ROC(self.y_test_100, y_predict_100, y_predict_200, y_predict_400, y_predict)
        self.plot_all_PR(self.y_test_100, y_predict_100, y_predict_200, y_predict_400, y_predict)

        thresholds = np.linspace(0.0, 1.0, 200)

        for t in thresholds:
            y_predict_binary = y_predict > t

            tn, fp, fn, tp = confusion_matrix(self.y_test_100, y_predict_binary).ravel()
            print('threshold=' + str(t))
            print(tn, fp, fn, tp)

        self.write_classification_toMat(y_predict, 0.3718)

    def plot_ROC(self, y_groundtruth, y_predict):
        fpr, tpr, _ = roc_curve(y_groundtruth, y_predict)
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

    def plot_PR(self, y_groundtruth, y_predict):
        precision, recall, _ = precision_recall_curve(y_groundtruth, y_predict)
        # APS = average_precision_score(y_groundtruth, y_predict)
        AUC = auc(recall, precision)
        print('the AUC is : %0.4f' % AUC)
        plt.figure()
        plt.plot(recall, precision, label='precision-recall curve (score = %0.4f)' % AUC)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_all_ROC(self, y_groundtruth, y_predict100, y_predict200, y_predict400, y_predict_combine):
        fpr100_GF_color, tpr100_GF_color, _ = roc_curve(y_groundtruth, y_predict100)
        fpr200_GF_color, tpr200_GF_color, _ = roc_curve(y_groundtruth, y_predict200)
        fpr400_GF_color, tpr400_GF_color, _ = roc_curve(y_groundtruth, y_predict400)
        fpr_combine_GF_color, tpr_combine_GF_color, _ = roc_curve(y_groundtruth, y_predict_combine)

        with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/decision_tree_ROC_GF_color.p3', 'rb') as f:
            fpr_tree_GF_color, tpr_tree_GF_color = pickle.load(f)

        # GF input
        with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/scale100_ROC_GF.p3', 'rb') as f:
            fpr100_GF, tpr100_GF = pickle.load(f)

        with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/scale200_ROC_GF.p3', 'rb') as f:
            fpr200_GF, tpr200_GF = pickle.load(f)

        with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/scale400_ROC_GF.p3', 'rb') as f:
            fpr400_GF, tpr400_GF = pickle.load(f)

        with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/multiscale_ROC_GF.p3', 'rb') as f:
            fpr_combine_GF, tpr_combine_GF = pickle.load(f)

        with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/decision_tree_ROC_GF.p3', 'rb') as f:
            fpr_tree_GF, tpr_tree_GF = pickle.load(f)
            
        AUC100 = auc(fpr100, tpr100)
        AUC200 = auc(fpr200, tpr200)
        AUC400 = auc(fpr400, tpr400)
        AUC_combine = auc(fpr_combine, tpr_combine)
        AUC_tree = auc(fpr_tree, tpr_tree)

        line_width = 3
        plt.figure()
        plt.plot(fpr100_GF_color, tpr100_GF_color, 'b-', linewidth=line_width, label='100x100x12 color+GF')
        plt.plot(fpr200_GF_color, tpr200_GF_color, 'g-', linewidth=line_width, label='200x200x24 color+GF')
        plt.plot(fpr400_GF_color, tpr400_GF_color, 'r-', linewidth=line_width, label='400x400x48 color+GF')
        plt.plot(fpr_combine_GF_color, tpr_combine_GF_color, 'c-', linewidth=line_width, label='Multi-scale CNN color+GF')
        plt.plot(fpr_combine_GF_color, tpr_combine_GF_color, 'c-', linewidth=line_width)

        plt.plot(fpr100_GF_color, tpr100_GF_color, 'b-', linewidth=line_width)
        plt.plot(fpr200_GF_color, tpr200_GF_color, 'g-', linewidth=line_width)
        plt.plot(fpr400_GF_color, tpr400_GF_color, 'r-', linewidth=line_width)
        plt.plot(fpr_combine_GF_color, tpr_combine_GF_color, 'c-', linewidth=line_width)


        plt.plot(fpr_tree_GF_color, tpr_tree_GF_color, 'm-', linewidth=line_width, label='Random forest color+GF')
        plt.plot(fpr_tree_GF_color, tpr_tree_GF_color, 'm-', linewidth=line_width)

        plt.plot(fpr100_GF, tpr100_GF, color='darkgray', linewidth=line_width, label='100x100x12 GF')
        plt.plot(fpr200_GF, tpr200_GF, color='chartreuse', linewidth=line_width, label='200x200x24 GF')
        plt.plot(fpr400_GF, tpr400_GF, color='darkorange', linewidth=line_width, label='400x400x48 GF')
        plt.plot(fpr_combine_GF, tpr_combine_GF, color='maroon', linewidth=line_width, label='Multi-scale CNN GF')
        plt.plot(fpr_tree_GF, tpr_tree_GF, color='burlywood', linewidth=line_width, label='Random forest GF')
        plt.plot(fpr_combine_GF, tpr_combine_GF, color='maroon', linewidth=line_width)
        plt.plot(fpr_tree_GF, tpr_tree_GF, color='burlywood', linewidth=line_width)
        plt.rcParams.update({'font.size': 18})

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_all_PR(self, y_groundtruth, y_predict100, y_predict200, y_predict400, y_predict_combine):
        precision100_GF_color, recall100_GF_color, _ = precision_recall_curve(y_groundtruth, y_predict100)
        precision200_GF_color, recall200_GF_color, _ = precision_recall_curve(y_groundtruth, y_predict200)
        precision400_GF_color, recall400_GF_color, _ = precision_recall_curve(y_groundtruth, y_predict400)
        precision_combine_GF_color, recall_combine_GF_color, _ = precision_recall_curve(y_groundtruth, y_predict_combine)

        with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/decision_tree_PR_GF_color.p3', 'rb') as f:
            precision_tree_GF_color,  recall_tree_GF_color = pickle.load(f)

        # GF input
        with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/scale100_PR_GF.p3', 'rb') as f:
            precision100_GF, recall100_GF = pickle.load(f)

        with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/scale200_PR_GF.p3', 'rb') as f:
            precision200_GF, recall200_GF = pickle.load(f)

        with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/scale400_PR_GF.p3', 'rb') as f:
            precision400_GF, recall400_GF = pickle.load(f)

        with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/multiscale_PR_GF.p3', 'rb') as f:
            precision_combine_GF, recall_combine_GF = pickle.load(f)

        with open('C:/Users/yxl1214/Dropbox/C6M3 GFP/metastases_segmentation/Python code/saved_results/saved_data/decision_tree_PR_GF.p3', 'rb') as f:
            precision_tree_GF,  recall_tree_GF = pickle.load(f)

        AUC100 = auc(recall100, precision100)
        AUC200 = auc(recall200, precision200)
        AUC400 = auc(recall400, precision400)
        AUC_combine = auc(recall_combine, precision_combine)
        AUC_tree = auc(recall_tree, precision_tree)

        line_width = 3
        plt.figure()
        plt.plot(recall100_GF_color, precision100_GF_color, 'b-', linewidth=line_width)
        plt.plot(recall200_GF_color, precision200_GF_color, 'g-', linewidth=line_width)
        plt.plot(recall400_GF_color, precision400_GF_color, 'r-', linewidth=line_width)
        plt.plot(recall_combine_GF_color, precision_combine_GF_color, 'c-', linewidth=line_width)
        plt.plot(recall_tree_GF_color, precision_tree_GF_color, 'm-', linewidth=line_width)

        plt.plot(recall100_GF, precision100_GF, color='darkgray', linewidth=line_width)
        plt.plot(recall200_GF, precision200_GF, color='chartreuse', linewidth=line_width)
        plt.plot(recall400_GF, precision400_GF, color='darkorange', linewidth=line_width)
        plt.plot(recall_combine_GF, precision_combine_GF, color='maroon', linewidth=line_width)
        plt.plot(recall_tree_GF, precision_tree_GF, color='burlywood', linewidth=line_width)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-recall Curve')
        plt.legend(loc="lower right")
        plt.show()

if __name__ == "__main__":
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

    analysis = FP_FN_analysis_multilevel(model100_path, model200_path, model400_path, small_mets=False)
    analysis.FP_FN_index_DL()
    
