##################################################################
#
#   CURIAL-Federated: A Federated Learning Pipeline for COVID-19 screening
#   Utils.py - a library of common functions and notes requires for the FL pipeline
#   Version: v1.0
#
##################################################################
from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, recall_score, precision_score, accuracy_score,roc_auc_score, confusion_matrix, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from scipy.special import ndtri
from math import sqrt
import pickle
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers

import select
import time
import sys
import os
import openml
from statsmodels.stats.contingency_tables import mcnemar

localpath = ''

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

#The training dataset is now ready - cases are now in 'cases', and controls are now in 'matched_controls' - stored in trainingDataSet
""" Return the relevant feature list  """
def featureList(featureSet):
    if featureSet == "OLO & Vitals":
        featureList = pd.Series(['Blood_Test BASOPHILS', 'Blood_Test EOSINOPHILS', 'Blood_Test HAEMATOCRIT', 'Blood_Test HAEMOGLOBIN', 'Blood_Test LYMPHOCYTES', 'Blood_Test MEAN CELL VOL.', 'Blood_Test MONOCYTES', 'Blood_Test NEUTROPHILS', 'Blood_Test PLATELETS', 'Blood_Test WHITE CELLS', 'Vital_Sign Respiratory Rate', 'Vital_Sign Heart Rate', 'Vital_Sign Systolic Blood Pressure', 'Vital_Sign Temperature Tympanic', 'Vital_Sign Oxygen Saturation', 'Vital_Sign Delivery device used', 'Vital_Sign Diastolic Blood Pressure'])
    elif featureSet == "Bloods & Vitals":
        featureList = pd.Series(['Blood_Test ALBUMIN', 'Blood_Test ALK.PHOSPHATASE', 'Blood_Test ALT', 'Blood_Test BASOPHILS', 'Blood_Test BILIRUBIN', 'Blood_Test CREATININE', 'Blood_Test CRP', 'Blood_Test EOSINOPHILS', 'Blood_Test HAEMATOCRIT', 'Blood_Test HAEMOGLOBIN', 'Blood_Test LYMPHOCYTES', 'Blood_Test MEAN CELL VOL.', 'Blood_Test MONOCYTES', 'Blood_Test NEUTROPHILS', 'Blood_Test PLATELETS', 'Blood_Test POTASSIUM', 'Blood_Test SODIUM', 'Blood_Test UREA', 'Blood_Test WHITE CELLS', 'Blood_Test eGFR', 'Vital_Sign Respiratory Rate', 'Vital_Sign Heart Rate', 'Vital_Sign Systolic Blood Pressure', 'Vital_Sign Temperature Tympanic', 'Vital_Sign Oxygen Saturation', 'Vital_Sign Delivery device used', 'Vital_Sign Diastolic Blood Pressure'])
    elif featureSet == "Bloods & Blood_Gas & Vitals":
        featureList = pd.Series(['Blood_Test ALBUMIN', 'Blood_Test ALK.PHOSPHATASE', 'Blood_Test ALT', 'Blood_Test APTT', 'Blood_Test BASOPHILS', 'Blood_Test BILIRUBIN', 'Blood_Test CREATININE', 'Blood_Test CRP', 'Blood_Test EOSINOPHILS', 'Blood_Test HAEMATOCRIT', 'Blood_Test HAEMOGLOBIN', 'Blood_Test LYMPHOCYTES', 'Blood_Test MEAN CELL VOL.', 'Blood_Test MONOCYTES', 'Blood_Test NEUTROPHILS', 'Blood_Test PLATELETS', 'Blood_Test POTASSIUM', 'Blood_Test Prothromb. Time', 'Blood_Test SODIUM', 'Blood_Test UREA', 'Blood_Test WHITE CELLS', 'Blood_Test eGFR', 'Blood_Gas BE Std (BG)', 'Blood_Gas Bicarb (BG)', 'Blood_Gas Ca+ + (BG)', 'Blood_Gas cLAC (BG)', 'Blood_Gas Glucose (BG)', 'Blood_Gas Hb (BG)', 'Blood_Gas Hct (BG)', 'Blood_Gas K+ (BG)', 'Blood_Gas Na+ (BG)', 'Blood_Gas O2 Sat (BG)', 'Blood_Gas pCO2 POC', 'Blood_Gas pO2 (BG)', 'Vital_Sign Respiratory Rate', 'Vital_Sign Heart Rate', 'Vital_Sign Systolic Blood Pressure', 'Vital_Sign Temperature Tympanic', 'Vital_Sign Oxygen Saturation', 'Vital_Sign Delivery device used', 'Vital_Sign Diastolic Blood Pressure'])
    return featureList

""" Standardize features """
def standardize_features(X, siteName):
    scaler = MinMaxScaler()
    scaler.fit(X)
    name = 'norm_' + siteName + '.pkl'
    with open(os.path.join(localpath,name),'wb') as f:
        pickle.dump(scaler,f)

    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    return X, scaler

#Deep Learning Model Class
class DNNModel(tf.keras.Model):
    def __init__(self,nodes, **kwargs):
        super().__init__()
        self.d1=layers.Dense(nodes,activation='relu')
        self.dropout1=tf.keras.layers.Dropout(0.5)
        self.d5=layers.Dense(1,activation='sigmoid')

    def forward(self, x,train=False):
        x = self.d1(x)
        x = self.dropout1(x,training=train)
        x = self.d5(x)
        return x

    def call(self, inputs):
        x = self.forward(inputs,train=False)
        return x

def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

""" Function no longer used - inheriting parameters from 10CV pre-training exercise"""
def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2  # MNIST has 10 classes
    n_features = 27  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

######### Model Evaluation functions
def calculateMetricsWithProbs (preds, probs, ground_truth, alpha, aucWithCIs=True):
        precision = precision_score(ground_truth,preds,zero_division=0)
        recallAchieved = recall_score(ground_truth,preds,zero_division=0)
        accuracy = accuracy_score(ground_truth,preds)
        #auc = roc_auc_score(ground_truth,pred_probs)
        #ns_fpr, ns_tpr, _ = roc_curve(ground_truth, pred_probs)
        tn, fp, fn, tp = confusion_matrix(ground_truth,preds).ravel()
        specificity = tn/(tn+fp)
        npv = tn/(fn+tn)

        #Work out AUROC using delong method (in function above)
        auc, auc_cov = delong_roc_variance(ground_truth,probs)
        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        auc_ci = np.around(stats.norm.ppf(lower_upper_q,loc=auc,scale=auc_std), decimals=3)

        #CIs for accuracy, PPV and NPV, Se, Sp
        alpha = 0.95
        z = -ndtri((1.0-alpha)/2)
        accuracy_ci = np.around(np.around(_proportion_confidence_interval(tp+tn, tp+tn+fp+fn, z), decimals=3)*100,decimals=1)
        ppv_ci = np.around(np.around(_proportion_confidence_interval(tp, tp+fp, z), decimals=3)*100,decimals=1)
        npv_ci = np.around(np.around(_proportion_confidence_interval(tn, tn+fn, z), decimals=3)*100,decimals=1)
        se_ci = np.around(np.around(_proportion_confidence_interval(tp, tp+fn, z), decimals=3)*100,decimals=1)
        sp_ci = np.around(np.around(_proportion_confidence_interval(tn, tn+fp, z), decimals=3)*100,decimals=1)

        #Generate strings
        accuracy_withCI = "%s%% (%s - %s)" % (np.round(accuracy*100, 1), accuracy_ci[0], accuracy_ci[1])
        ppv_withCI =  "%s%% (%s - %s)" % (np.round(precision*100, 1), ppv_ci[0], ppv_ci[1])
        npv_withCI =  "%s%% (%s - %s)" % (np.round(npv*100, 1), npv_ci[0], npv_ci[1])
        se_withCI =  "%s%% (%s - %s)" % (np.round(recallAchieved*100, 1), se_ci[0], se_ci[1])
        sp_withCI =  "%s%% (%s - %s)" % (np.round(specificity*100, 1), sp_ci[0], sp_ci[1])
        roc_withCI =  "%s (%s - %s)" % (np.round(auc, 3), auc_ci[0], auc_ci[1])

        #Generate ROC curves
        fpr, tpr, thresholds = roc_curve(ground_truth, probs)

        f1score = 2*(precision*recallAchieved)/(precision+recallAchieved)

        #Do not include the CIs if aucWithCIs is false (added for benefit for a test set)
        if (aucWithCIs == False):
            roc_withCI = auc

        return se_withCI,sp_withCI,accuracy_withCI,roc_withCI,ppv_withCI,npv_withCI,fn,fp,f1score, fpr, tpr

######### Functions to work out 95% AUC_CIs
#AUC 95% CIs using DeLong's Method
## Adapted from https://gist.github.com/RaulSanchezVazquez/d338c271ace3e218d83e3cb6400a769c
def AUC_CIs (probs, ground_truth, alpha = 0.95):
        #Work out AUROC using delong method (in function above)
        auc, auc_cov = delong_roc_variance(ground_truth,probs)
        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        auc_ci = np.around(stats.norm.ppf(lower_upper_q,loc=auc,scale=auc_std), decimals=3)

        #Generate strings
        roc_withCI =  "%s (%s - %s)" % (np.round(auc, 3), auc_ci[0], auc_ci[1])

        return roc_withCI


# Source: https://gist.github.com/maidens/29939b3383a5e57935491303cf0d8e0b
# #
#     References
#     ----------
#     [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
#     with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman,
#     D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000.
# #
def _proportion_confidence_interval(r, n, z):
    """Compute confidence interval for a proportion.

    Follows notation described on pages 46--47 of [1].

    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman,
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000.
    """

    A = 2*r + z**2
    B = z*sqrt(z**2 + 4*r*(1 - r/n))
    C = 2*(n + z**2)
    return ((A-B)/C, (A+B)/C)

def sensitivity_and_specificity_with_confidence_intervals(TP, FP, FN, TN, alpha=0.95):
    """Compute confidence intervals for sensitivity and specificity using Wilson's method.

    This method does not rely on a normal approximation and results in accurate
    confidence intervals even for small sample sizes.

    Parameters
    ----------
    TP : int
        Number of true positives
    FP : int
        Number of false positives
    FN : int
        Number of false negatives
    TN : int
        Number of true negatives
    alpha : float, optional
        Desired confidence. Defaults to 0.95, which yields a 95% confidence interval.

    Returns
    -------
    sensitivity_point_estimate : float
        Numerical estimate of the test sensitivity
    specificity_point_estimate : float
        Numerical estimate of the test specificity
    sensitivity_confidence_interval : Tuple (float, float)
        Lower and upper bounds on the alpha confidence interval for sensitivity
    specificity_confidence_interval
        Lower and upper bounds on the alpha confidence interval for specificity

    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman,
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000.
    [2] E. B. Wilson, Probable inference, the law of succession, and statistical inference,
    J Am Stat Assoc 22:209-12, 1927.
    """

    #
    z = -ndtri((1.0-alpha)/2)

    # Compute sensitivity using method described in [1]
    sensitivity_point_estimate = TP/(TP + FN)
    sensitivity_confidence_interval = _proportion_confidence_interval(TP, TP + FN, z)

    # Compute specificity using method described in [1]
    specificity_point_estimate = TN/(TN + FP)
    specificity_confidence_interval = _proportion_confidence_interval(TN, TN + FP, z)

    return sensitivity_point_estimate, specificity_point_estimate, sensitivity_confidence_interval, specificity_confidence_interval

#### Function to calculate AUROC with 95% CIs
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:06:52 2018

@author: yandexdataschool

Original Code found in:
https://github.com/yandexdataschool/roc_comparison

updated: Raul Sanchez-Vazquez
"""

import numpy as np
import scipy.stats
from scipy import stats

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight

def compute_ground_truth_statistics_no_weight(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics_no_weight(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)

    #NB: The calc_pvalues function returns log(10) of pvalues, and therefore this needs to be reversed prior to returning
    return np.array2string((10**calc_pvalue(aucs, delongcov))[0][0])

def input_with_timeout(prompt, timeout):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [],[], timeout)
    if ready:
        return sys.stdin.readline().rstrip('\n') # expect stdin to be line-buffered
    raise TimeoutExpired

#A function to perform the McNemars test between local and current model predictions
def compareModels(ground_truth, local_model_predictions, current_model_predictions):
    #Define contingency table
    bothCorrect = sum((local_model_predictions == ground_truth) & (current_model_predictions == ground_truth))
    localIncCurrCorr = sum((local_model_predictions != ground_truth) & (current_model_predictions == ground_truth))
    localCorrCurrInc = sum((local_model_predictions == ground_truth) & (current_model_predictions != ground_truth))
    bothIncorrect = sum((local_model_predictions != ground_truth) & (current_model_predictions != ground_truth))
    table = [[bothCorrect, localCorrCurrInc], [localIncCurrCorr, bothIncorrect]]

    # calculate mcnemar test
    result = mcnemar(table, exact=True)

    #Return pvalue
    return result.pvalue
