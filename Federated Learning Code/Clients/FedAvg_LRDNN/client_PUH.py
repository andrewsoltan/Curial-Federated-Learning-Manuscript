##################################################################
#
#   CURIAL-Federated: A Federated Learning Pipeline for COVID-19 screening
#   Version: v1.0
#
##################################################################

print ("Loading CURIAL Federated Learning Client, please wait")

import sys
import warnings
import flwr as fl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List, Tuple, Optional
import openpyxl
#import shap

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, recall_score, precision_score, accuracy_score,roc_auc_score, confusion_matrix, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.special import ndtri
from math import sqrt
import pickle
from statsmodels.stats.contingency_tables import mcnemar

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model

from tqdm import tqdm
import json

from copy import deepcopy
import random
import os
import time

import utils

##############################################
# Global configuration: FL server parameters #
##############################################
hostName = "4.231.101.5"
sslEnabled=False #set True for deployment
#hostName = "localhost"
port = "8002"
fastStart = True #Set to False on new
pauseTime = 10 #Set time pause to wait after completing LR and running DNN
serverAddress = hostName+":"+port
relaunchOnCompletion = True

##############################################
# SITE-SPECIFIC CONFIGURATION #
##############################################
global siteName
siteName = "PUH"
trainingOnly = False;

##The PUH Dataset is a Pre-pandemic file and a Pandemic file
#Import and pre-process both, followed by validation split
pathToPandemicData = "../raw/COVID_era_adms.csv"
pathToPrePandemicData = "../raw/Pre_COVID_era_adms.csv"
pathToPCR48hWindowFile = '../raw/pcr_result48.xlsx'
trainingCutOffDate = '2020-10-29'
validationFromDate = '2020-11-01'

#Import pre_processing for site-specific dataset from raw file
#Set present working directory to the true file location (to permit shortcut launching on Linux)
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(1, '../preproc')
sys.path.insert(2, '..')
import preprocess_PUH as preprocessor

##############################
# Analysis constants #
##############################
featureSet = "Bloods & Vitals" #, 'OLO & Vitals', 'Bloods & Blood_Gas & Vitals'];
imputationMethod = "median"
match_number = 10 #Match number for controls to cases during training
localpath = ''
resultsPath = 'Results/'

##############################
# Deep learning parameters #
##############################
tf.keras.backend.set_floatx('float32')
seeds=[10,20,100,223]
tf.random.set_seed(seeds[1])

##############################
# Evaluation Metrics #
##############################
metric_names = ['Recall','Precision','F1-Score','Accuracy','Specificity','PPV','NPV','AUC']
metrics = [recall_score,precision_score,accuracy_score,roc_auc_score]#,confusion_matrix]
metrics_dict = dict(zip(metric_names,metrics))
alpha = 0.95

if __name__ == "__main__":

    ##############
    #Stage 1: Run pre-processing script from raw dataset
    print ('Importing and pre-processing training datasets')
    #PUH has data divided in to pre-pandemic and pandemic files, we should combine these first to allow the standard pipeline to be followed

    #First, load pandemic data for PUH, specifying that the 48h window file should be replaced
    pandemicData = preprocessor.preprocess (pathToPandemicData, pathToPCR48hWindowFile,"Pandemic")
    prePandemicData = preprocessor.preprocess(pathToPrePandemicData,False,"Prepandemic")
    fullDataset = pd.concat([prePandemicData,pandemicData], ignore_index=True)

    ##############
    #Stage 2: Formation of correct training & validation cohorts (where applicable), noramlisation & imputation

    #Generate the Training set with reference to Cut Off Date
    preprocessedTrainingDataset = fullDataset[fullDataset.ArrivalDateTime <= trainingCutOffDate]

    #Restrict Population to patients admitted to Hospital
    preprocessedTrainingDataset = preprocessedTrainingDataset[(preprocessedTrainingDataset.Admission == 1.0) | (preprocessedTrainingDataset.ICU == 1.0)]

    #Perform split in to train and validation cohorts where applicable
    if trainingOnly:
        trainingDataset = preprocessedTrainingDataset
    elif not trainingOnly:
        #Perform Test/Validate split if not in same file
        #Or alternatively, load in separate file and pre-process
        trainingDataset = preprocessedTrainingDataset
        validationDataset = fullDataset[fullDataset.ArrivalDateTime >= validationFromDate]
        #Restrict validation to patients who had a Covid-19 test result
        validationDataset = validationDataset[(validationDataset['Covid-19 Positive'] == 1.0) | (validationDataset['Covid-19 Positive'] == 0.0)]

    print ("Training set: Covid-19 Cases " + str(trainingDataset['Covid-19 Positive'].sum()))
    print ("Training set: Total length " + str(trainingDataset.shape[0]))
    print ("Validation set: Covid-19 Cases " + str(validationDataset['Covid-19 Positive'].sum()))
    print ("Validation set: Total length " + str(validationDataset.shape[0]))

    print ('Preparing training cohorts')


    #Generate the full training set; with cases and matched controls
    """ Obtain Case Cohort """
    def load_case_data(data):
        condition = data['Covid-19 Positive']==True
        case_indices = data.index[condition]
        case_data = data.loc[case_indices,:]
        return case_data,case_indices

    """ Obtain Whole Control Cohort """
    def load_control_data(data):
        condition = data['ArrivalDateTime'] < '2019-12-01'
        control_indicies = data.index[condition]
        control_data = data.loc[control_indicies,:]
        return control_data,control_indicies

    """ Obtain Matched Control Cohort, Matched for Age (+/- 4 years), Gender and Ethnicity """
    def load_matched_control_cohort(match_number,control_data,data,case_indices):
        matched_cohort_indices = []
        for index in case_indices:
            patient_data = data.loc[index,:]
            patient_age = patient_data['Age']
            gender = patient_data['Gender']
            ethnicity = patient_data['Ethnicity']

            age_condition1 = control_data['Age'] < patient_age + 4
            age_condition2 = control_data['Age'] > patient_age - 4
            gender_condition = control_data['Gender'] == gender
            ethnicity_condition = control_data['Ethnicity'] == ethnicity

            matched_indices = control_data.index[age_condition1 & age_condition2 & gender_condition & ethnicity_condition]
            matched_indices = matched_indices.tolist()
            random.seed(0)
            matched_indices = random.sample(matched_indices,len(matched_indices))

            valid_indices = [index for index in matched_indices if index not in matched_cohort_indices][:match_number]
            matched_cohort_indices.extend(valid_indices)
            control_cohort = control_data.loc[matched_cohort_indices,:] #index name not location

        return control_cohort

    """ Combine cases & control cohorts """
    def load_combined_cohort(cases,controls):
        df = pd.concat((cases, controls),0)
        return df

    #Quickstart File name
    name = 'quickstart_' + siteName + ' Training Set.pkl'

    if (not os.path.exists(name)) or (not fastStart):
        #Generate cohorts of cases, controls, and matched controls
        cases, case_indices = load_case_data (trainingDataset)
        controls, control_indicies = load_control_data (trainingDataset)
        matched_controls = load_matched_control_cohort(match_number,controls,trainingDataset,case_indices)

        #Combine matched controls with cases to give a full training set
        caseControlTrainingSet = load_combined_cohort(cases,matched_controls)

        with open(os.path.join(localpath,name),'wb') as f:
            pickle.dump(caseControlTrainingSet,f)
    elif fastStart and (os.path.exists(name)):
        caseControlTrainingSet = pd.read_pickle(name)

    """ Generate X & Y for training  """
    def load_data_splits(df,featureList):
            X = df[featureList]
            Y = df['Covid-19 Positive']
            return X,Y

    #Get Features, generate X & Y
    features = utils.featureList(featureSet)
    X,Y = load_data_splits(caseControlTrainingSet, features)

    #Calculate training population median values for sending to server
    trainingPopulationMedianValues = json.dumps(X.median().to_dict())

    #Now impute missing features, and perform standardization
    #For the purposes of federated evaluation, the imputer and scaler are stored in imputer and scaler

    """ Train imputer on training set """
    def impute_missing_features(imputation_method,X,siteName):
        if imputation_method in ['mean','median','MICE']:
            if imputation_method == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif imputation_method == 'median':
                imputer = SimpleImputer(strategy='median')

            imputer.fit(X)
            X = pd.DataFrame(imputer.transform(X), columns=X.columns)

        name = 'imputer_' + siteName + '_'+imputation_method+'.pkl'
        with open(os.path.join(localpath,name),'wb') as f:
            pickle.dump(imputer,f)
        return X, imputer

    #Perform imputation & scaling
    X, imputer = impute_missing_features(imputationMethod,X,siteName)

    #Perform scaling
    X, scaler = utils.standardize_features(X,siteName)

    #Fill Y missing values with 0, as these are all true negatives
    Y[np.isnan(Y)] = 0

    #Save X and break script
    #pd.DataFrame(X, columns=X.columns).to_csv('X Dump OUH.csv')
    #exit()

    #Now perform an 80/20 train/test split using the training dataset (with the 20% being used to calibrate)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #Prepare validation set where validation is being performed
    if not trainingOnly:
        X_val, Y_val = load_data_splits(validationDataset, features)
        X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    elif trainingOnly:
        #If not an evaluation site, use test set as the validation set
        X_val = X_test
        Y_val = Y_test

    ##############
    #Stage 3: Define LR model & threshold selection function
    #K fold cross validation has been disabled
    """ Function to determine the correct model threshold to achieve a given metric"""
    def find_threshold_at_metric(model,inputs,outputs,best_threshold,metric_of_interest,value_of_interest,results_df,fold_number,error, match_number, modelType = "LR"):
        ground_truth = outputs['eval']

        """ Probability Values for Predictions """
        if modelType == "LR":
            probs = model.predict_proba(inputs['eval'])[:,1]
        elif modelType == "DNN":
            probs = model.predict_on_batch(inputs['eval'].values).ravel()

        threshold_metrics = pd.DataFrame(np.zeros((500,8)),index=np.linspace(0,1,500),columns=['Recall','Precision','F1-Score','Accuracy','Specificity','PPV','NPV','AUC'])
        prev = 1/(match_number+1)
        for t in np.linspace(0,1,500):
            preds = np.where(probs>t,1,0)
            recall = recall_score(ground_truth,preds,zero_division=0)
            accuracy = accuracy_score(ground_truth,preds)
            auc = roc_auc_score(ground_truth,probs)
            tn, fp, fn, tp = confusion_matrix(ground_truth,preds).ravel()
            specificity = tn/(tn+fp)
            ppv = (recall* (prev))/(recall * prev + (1-specificity) * (1-prev))
            precision = ppv
            f1score = 2*(precision*recall)/(precision+recall)
            if tn== 0 and fn==0:
                npv = 0
            else:
                npv = (specificity* (1-prev))/(specificity * (1-prev) + (1-recall) * (prev))
            threshold_metrics.loc[t,['Recall','Precision','F1-Score','Accuracy','Specificity','PPV','NPV','AUC']] = [recall,precision,f1score,accuracy,specificity,ppv,npv,auc]

        """ Identify Results that Satisfy Constraints and Best Threshold """
     #   value_of_interest = threshold_metrics.loc[:,metric_of_interest].max()
        condition1 = threshold_metrics.loc[:,metric_of_interest] < value_of_interest + error
        condition2 = threshold_metrics.loc[:,metric_of_interest] > value_of_interest - error
        combined_condition = condition1 & condition2
        if metric_of_interest == 'Recall':
            sort_col = 'Precision'
        elif metric_of_interest == 'Precision':
            sort_col = 'Recall'
        elif metric_of_interest == 'F1-Score':
            sort_col = 'F1-Score'
        sorted_results = threshold_metrics[combined_condition].sort_values(by=sort_col,ascending=False)
      #  print(sorted_results)
        if len(sorted_results) > 0:
            """ Only Record Value if Condition is Satisfied """
            results_df.loc[['Recall','Precision','F1-Score','Accuracy','Specificity','PPV','NPV','AUC'],fold_number] = sorted_results.iloc[0,:]
            best_threshold.iloc[fold_number] = sorted_results.iloc[0,:].name
        else:
            print('No Threshold Found for Constraint!')

        return best_threshold, results_df

    while relaunchOnCompletion == True:
        #Create Ouptut Dataframe for Prospective Evaluation Matrix
        GlobalModelFedEvalResults = pd.DataFrame(columns=['Site ID','Model Type', 'Iteration', 'CURIAL Targetted Recall','Imputation','Achieved Sensitivity','Specificity','Accuracy','AUROC','Precision','NPV','n False Negative','n False Positive','F1','DeLong P Current vs Global', 'DeLong LR vs DNN'])
        SiteSpecificModelFedEvalResults = pd.DataFrame(columns=['Site ID','Model Type', 'Iteration', 'CURIAL Targetted Recall','Imputation','Achieved Sensitivity','Specificity','Accuracy','AUROC','Precision','NPV','n False Negative','n False Positive','F1', 'DeLong P Current vs Global', 'DeLong LR vs DNN'])
        TestSetResults = pd.DataFrame(columns=['Site ID','Model Type', 'Iteration', 'CURIAL Targetted Recall','Imputation','Achieved Sensitivity','Specificity','Accuracy','AUROC','Precision','NPV','n False Negative','n False Positive','F1','DeLong P Current vs Global', 'DeLong LR vs DNN'])
        LRValidationSetPreds = np.zeros(1)
        ##############
        """ Define Logistic Regression Federated Flower Client & fit function"""
        class LRClient(fl.client.NumPyClient):
            iteration = 0
            modelType = "LR"
            global siteName


            def get_parameters(self, config):  # type: ignore
                return utils.get_model_parameters(model)


            """Fit Model locally  & send back parameters"""
            def fit(self, parameters, config):  # type: ignore
                print(f"Round {config['server_round']}")

                #Except on the 0th iteration, update with weights from server
                if self.iteration > 0:
                    utils.set_model_params(model, parameters)

                """ Fit model """
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    #Set max iterations
                    model.set_params(**{'max_iter':config['max_iterations']})
                    model.fit(X_train, Y_train)

                # """ Perform SHAP analysis for the 0th iteration - i.e. model training on single site - and save file """
                # import shap
                # if self.iteration == 0:
                #     print ("Performing explanability analysis on locally-trained model")
                #     name = 'Results/shapExplanations_LR_' + siteName + ' 0th iteration.pkl'
                #     # shapExplainer = shap.Explainer(model.predict, X_train, feature_names=X_train.columns)
                #     # generateExplainer = shapExplainer(X_test)
                #     ind=np.where(Y_train==1)[0]
                #     shapExplainer = shap.LinearExplainer(model, X_train.iloc[ind])
                #     ind=np.where(Y_test==1)[0]
                #     generateExplainer = shapExplainer.shap_values(X_test.iloc[ind])
                #
                #     #Save explainer item
                #     with open(name, "wb") as f:
                #         pickle.dump(generateExplainer,f)

                #Update iteration number
                self.iteration = self.iteration + 1

                return utils.get_model_parameters(model), len(X_train), {'trainingPopulationMedian':trainingPopulationMedianValues}

            ##############
            #Stage 5: Federated evaluation on prospective/true validation set
            """ Perform Calibration on the Test set, assessment on the Test Set, followed by Federated Evaluation of the global model on the validation set """
            def evaluate(self, parameters, config):  # type: ignore

                #Get parameters of Global Model
                utils.set_model_params(model, parameters)

                #If the client successfully runs, store the last successful IP as the new default IP
                originalServerAddress = serverAddress

                """ Determine optimal threshold first by fitting on test set """
                results_df = pd.DataFrame(np.zeros((len(metric_names),1)),index=metric_names)
                best_threshold = pd.Series(np.zeros((1)))
                inputs,outputs = dict(), dict()
                inputs['eval'], outputs['eval'] = [X_test,Y_test]
                thresholds_on_test, results_df = find_threshold_at_metric(model, inputs, outputs, best_threshold, "Recall", config['targetSensitivity'], results_df, 0, 0.04, match_number, self.modelType)
                thresholdOnTestSet = thresholds_on_test.mean()

                """ Calculate performance on Test Set, using threshold from calibration"""
                predsOnTestSet=model.predict_proba(X_test)[:,1]
                predictedLabels = np.where(predsOnTestSet>thresholdOnTestSet,1,0)
                recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, fpr, tpr = utils.calculateMetricsWithProbs (predictedLabels, predsOnTestSet, Y_test, alpha)
                #Save local model predictions on TEST SET to a global variable
                if config['server_round'] == 1:
                    self.localModelPredictionsTestSet = predictedLabels
                #Calculate DeLongP value for current model versus the Local Model
                deLongPValue = utils.delong_roc_test(Y_test.values, self.localModelPredictionsTestSet, predictedLabels)
                TestSetResults.loc[self.iteration,:] = [siteName, modelType, self.iteration, config['targetSensitivity'],imputationMethod,recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, deLongPValue, np.nan]
                #Encode the test set results in JSON, and transmit to central server
                TestSet_PerformanceResults_dict = json.dumps(TestSetResults.loc[self.iteration,:].to_dict())

                """ Perform federated evaluation on validation set (NB: If no validation set, we will use the test set) """
                predsOnValidationSet=model.predict_proba(X_val)[:,1]
                loss = log_loss(Y_val,predsOnValidationSet)

                #Fully evaluate on the prospective set with 95% CIs
                #First, generated predicted labels based on selected threshold set during testing
                predictedLabels = np.where(predsOnValidationSet>thresholdOnTestSet,1,0)

                """If evaluating the local model, save local model predictions for future use calculating P Values"""
                if config['server_round'] == 1:
                    self.localModelPredictionsValidationSet = predictedLabels

                """ Perform DeLong's Testing to compare local model with current global model """
                #NB: For the local model, the p value will be 1.0
                deLongPValue = utils.delong_roc_test(Y_val.values, self.localModelPredictionsValidationSet, predictedLabels)

                """Set a global variable with the LR Validation Set Preds to allow later adjustment"""
                global LRValidationSetPreds
                LRValidationSetPreds = predictedLabels

                #Now fully evaluate at the threshold chosen using validation set
                #Define the server round (if initial eval, set round to 0)
                recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, fpr, tpr = utils.calculateMetricsWithProbs (predictedLabels, predsOnValidationSet, Y_val, alpha)
                GlobalModelFedEvalResults.loc[self.iteration,:] = [siteName, modelType, self.iteration, config['targetSensitivity'],imputationMethod,recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score,deLongPValue, np.nan]

                print (auc)
                #Encode the federated evaluation results in JSON, and transmit to central server
                GlobalModelResults_dict = json.dumps(GlobalModelFedEvalResults.loc[self.iteration,:].to_dict())

                """ Calculate variables for a ROC curve on validation set """
                #rocDf=pd.DataFrame()
                testset_fpr, testset_tpr, _ = roc_curve(Y_val, predsOnValidationSet,drop_intermediate=True)
                # rocDf['TPR']=testset_tpr
                # rocDf['FPR']=testset_fpr
                # rocDf['Site ID'] = siteName
                rocAnalysisOnFinalIter = pd.concat([pd.Series(testset_fpr),pd.Series(testset_tpr)], axis=1)
                rocAnalysisOnFinalIter = rocAnalysisOnFinalIter.rename(columns= {0: "FPR", 1: "TPR"})
                rocAnalysisOnFinalIter["Site ID"] = siteName
                rocAnalysisDict = json.dumps(rocAnalysisOnFinalIter.to_dict())

                #Save as CSV - temporary
                #rocDf.to_csv("Oxford ROC Round 1.csv")

                """Perform fine-tuning & evaluation of site-specific model """
                #Now perform prospective evaluation for a fine-tuned model
                siteSpecificModelResults = self.evaluateFineTunedModel(parameters, config, predictedLabels)

                #Message threshold back to server
                return loss, len(X_val), { "Threshold":thresholdOnTestSet , "AUROC": auc, "ROC_analysis": rocAnalysisDict, "Test_Set_Iter_Results": TestSet_PerformanceResults_dict, "Global_Model_Iter_Results": GlobalModelResults_dict, "Site_Personalised_Iter_Results": siteSpecificModelResults}

            """ Evaluate a fine-tuned, site-personalised model """
            def evaluateFineTunedModel(self, parameters, config, global_model_predictions):

                #Create local copy of model
                siteTunedModel = model

                #Perform a local model update
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    siteTunedModel.fit(X_train, Y_train)

                """ Determine optimal threshold first by fitting on test set """
                results_df = pd.DataFrame(np.zeros((len(metric_names),1)),index=metric_names)
                best_threshold = pd.Series(np.zeros((1)))
                inputs,outputs = dict(), dict()
                inputs['eval'], outputs['eval'] = [X_test,Y_test]
                thresholds_on_test, results_df = find_threshold_at_metric(siteTunedModel, inputs, outputs, best_threshold, "Recall", config['targetSensitivity'], results_df, 0, 0.04, match_number, self.modelType)
                thresholdOnTestSet = thresholds_on_test.mean()

                """ Perform federated evaluation on validation set (NB: If no validation set, we will use the test set) """
                predsOnValidationSet=siteTunedModel.predict_proba(X_val)[:,1]
                loss = log_loss(Y_val,predsOnValidationSet)

                #Fully evaluate on the prospective set with 95% CIs
                #First, generated predicted labels based on selected threshold set during testing
                predictedLabels = np.where(predsOnValidationSet>thresholdOnTestSet,1,0)

                #Compare to global mdoel
                deLongPValue = utils.delong_roc_test(Y_val.values, global_model_predictions, predictedLabels)

                #Now fully evaluate at the threshold chosen using held out test set
                #Define the server round (if initial eval, set round to 0)
                recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, fpr, tpr = utils.calculateMetricsWithProbs (predictedLabels, predsOnValidationSet, Y_val, alpha)
                SiteSpecificModelFedEvalResults.loc[self.iteration,:] = [siteName, modelType, self.iteration, config['targetSensitivity'],imputationMethod,recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, deLongPValue, np.nan]

                #Encode the results in JSON, and transmit to central server
                #Make a dictionary of results, to transmit back to server
                siteSpecificModelResults = json.dumps(SiteSpecificModelFedEvalResults.loc[self.iteration,:].to_dict())

                return siteSpecificModelResults

                #Message threshold back to server
                #return loss, len(X_val), { "Threshold":thresholdOnTestSet , "AUROC": auc, "Site_Personalised_Iter_Results": siteSpecificModelResults}


        ##############
        #Stage 4B: Deep Neural network - define Federated Flower Client
        class DNNClient(fl.client.NumPyClient):
            modelType = "DNN"
            iteration = 0
            localModelPredictionsValidationSet = []
            global siteName

            #On intialisation of the class, set an offset to the number of iterations to keep the tracking on output variable
            def __init__(self):
                self.iteration = self.iteration + GlobalModelFedEvalResults.shape[0]

            def fit(self, parameters, config):
                print(f"Round {config['server_round']}")

                #Update parameters with global params, with exception of the 0th iteration
                if self.iteration > GlobalModelFedEvalResults.shape[0]:
                    model.set_weights(parameters)

                es = EarlyStopping(monitor='val_auc', mode='min', verbose=1, patience=config['patience'])
                model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=localpath+'local_models/'+siteName+'.h5', save_weights_only=True, monitor='val_auc',mode='max',save_best_only=True)
                model.fit(X_train.values, Y_train.values, validation_data=(X_test.values,Y_test.values), epochs=config['local_epochs'], batch_size=config['batch_size'],verbose=1, shuffle=True, callbacks=[model_checkpoint_callback,es])

                #Load best performing local model saved in callback
                model.load_weights(localpath+'local_models/'+siteName+'.h5')

                # """ Perform SHAP analysis for the 0th iteration - i.e. model training on single site - and save file """
                # if self.iteration == GlobalModelFedEvalResults.shape[0]:
                #     print ("Performing explanability analysis on locally-trained model")
                #     name = 'Results/shapExplanations_DNN_' + siteName + ' 0th iteration.pkl'
                #     tf.compat.v1.disable_v2_behavior()
                #     ind=np.where(Y_train==1)[0]
                #     #shapExplainer = shap.DeepExplainer(model, X_train.iloc[ind])
                #     shapExplainer = shap.DeepExplainer(model, X_train.iloc[ind])
                #
                #     ind=np.where(Y_test==1)[0]
                #     generateExplainer = shapExplainer.shap_values(X_test.iloc[ind].values)
                #     tf.compat.v1.enable_v2_behavior()
                    #
                    # #Save explainer item
                    # with open(name, "wb") as f:
                    #     pickle.dump(generateExplainer,f)

                #Increment Iteration
                self.iteration = self.iteration + 1

                return model.get_weights(), len(X_train), {'trainingPopulationMedian':trainingPopulationMedianValues}

            """ Evaluate Global Model on Prospective Validation Set """
            def evaluate(self, parameters, config):

                """" Update model with global parameters """
                model.set_weights(parameters)

                """ Determine optimal threshold first by fitting on test set """
                results_df = pd.DataFrame(np.zeros((len(metric_names),1)),index=metric_names)
                best_threshold = pd.Series(np.zeros((1)))
                inputs,outputs = dict(), dict()
                inputs['eval'], outputs['eval'] = [X_test,Y_test]
                thresholds_on_test, results_df = find_threshold_at_metric(model, inputs, outputs, best_threshold, "Recall", config['targetSensitivity'], results_df, 0, 0.04, match_number, self.modelType)
                thresholdOnTestSet = thresholds_on_test.mean()

                """ Calculate performance on Test Set, using threshold from calibration on test set"""
                predsOnTestSet=model.predict_on_batch(X_test.values).ravel()
                predictedLabels = np.where(predsOnTestSet>thresholdOnTestSet,1,0)
                #Save local model predictions on TEST SET to a global variable
                if config['server_round'] == 1:
                    self.localModelPredictionsTestSet = predictedLabels
                #Calculate DeLongP value for current model versus the Local Model
                deLongPValue = utils.delong_roc_test(Y_test.values, self.localModelPredictionsTestSet, predictedLabels)
                recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, fpr, tpr = utils.calculateMetricsWithProbs (predictedLabels, predsOnTestSet, Y_test, alpha)
                TestSetResults.loc[self.iteration,:] = [siteName, modelType, self.iteration, config['targetSensitivity'],imputationMethod,recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, deLongPValue, np.nan]
                #Encode the test set results in JSON, and transmit to central server
                TestSet_PerformanceResults_dict = json.dumps(TestSetResults.loc[self.iteration,:].to_dict())

                """ Perform federated evaluation on validation set (NB: If no validation set, we will use the test set) """
                predsOnValidationSet=model.predict_on_batch(X_val.values).ravel()
                loss = log_loss(Y_val,predsOnValidationSet)

                #Fully evaluate on the prospective set with 95% CIs
                #First, generated predicted labels based on selected threshold
                predictedLabels = np.where(predsOnValidationSet>thresholdOnTestSet,1,0)

                """If evaluating the local model, save local model predictions for future use"""
                if config['server_round'] == 1:
                    self.localModelPredictionsValidationSet = predictedLabels

                """ Perform DeLong's Testing to compare local model with current global model """
                #NB: For the local model, the p value will be 1.0
                deLongPValue = utils.delong_roc_test(Y_val.values, self.localModelPredictionsValidationSet, predictedLabels)

                """ Perform DeLong's Testing to compare current DNN model with the saved global LR predictions """
                global LRValidationSetPreds
                LRvsDNNDeLongP = utils.delong_roc_test(Y_val.values, LRValidationSetPreds, predictedLabels)

                #Now fully evaluate at the threshold chosen using held out test set
                #Define the server round (if initial eval, set round to 0)
                recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, fpr, tpr = utils.calculateMetricsWithProbs (predictedLabels, predsOnValidationSet, Y_val, alpha)
                GlobalModelFedEvalResults.loc[self.iteration,:] = [siteName, modelType, self.iteration, config['targetSensitivity'],imputationMethod,recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score,deLongPValue, LRvsDNNDeLongP]
                print (auc)

                #Encode the results in JSON, and transmit to central server
                #Make a dictionary of results, to transmit back to server
                GlobalModelResults_dict = json.dumps(GlobalModelFedEvalResults.loc[self.iteration,:].to_dict())

                """ Calculate variables for a ROC curve on validation set -- transmit to server """
                testset_fpr, testset_tpr, _ = roc_curve(Y_val, predsOnValidationSet)
                rocAnalysisOnFinalIter = pd.concat([pd.Series(testset_fpr),pd.Series(testset_tpr)], axis=1)
                rocAnalysisOnFinalIter = rocAnalysisOnFinalIter.rename(columns= {0: "FPR", 1: "TPR"})
                rocAnalysisOnFinalIter["Site ID"] = siteName
                rocAnalysisDict = json.dumps(rocAnalysisOnFinalIter.to_dict())

                #print (config)
                """Perform fine-tuning & evaluation of site-specific model """
                siteSpecificModelResults = self.evaluateFineTunedModel(parameters, config, predictedLabels)

                #Message results back to server
                return loss, len(X_val), { "Threshold":thresholdOnTestSet , "AUROC": auc, "ROC_analysis": rocAnalysisDict, "Test_Set_Iter_Results": TestSet_PerformanceResults_dict,  "Global_Model_Iter_Results": GlobalModelResults_dict, "Site_Personalised_Iter_Results": siteSpecificModelResults}

            """ Evaluate a fine-tuned, personalised model """
            def evaluateFineTunedModel(self, parameters, config, global_model_predictions):
                #print (config)

                #Perform a fine-tuning step to make a site-specific ('personalised federated') model
                esFT = EarlyStopping(monitor='val_auc', mode='min', verbose=1, patience=config['patience'])
                #model_checkpoint_callbackFT = tf.keras.callbacks.ModelCheckpoint(filepath=localpath+'local_models/'+siteName+'.h5', save_weights_only=True, monitor='val_auc',mode='max',save_best_only=True)
                siteTunedModel = model
                siteTunedModel.fit(X_train.values, Y_train.values, validation_data=(X_test.values,Y_test.values), epochs=config['local_epochs'], batch_size=config['batch_size'],verbose=1, shuffle=True, callbacks=[esFT])

                """ Determine optimal threshold of fine tuned model first by fitting on test set """
                results_df = pd.DataFrame(np.zeros((len(metric_names),1)),index=metric_names)
                best_threshold = pd.Series(np.zeros((1)))
                inputs,outputs = dict(), dict()
                inputs['eval'], outputs['eval'] = [X_test,Y_test]
                thresholds_on_test, results_df = find_threshold_at_metric(siteTunedModel, inputs, outputs, best_threshold, "Recall", config['targetSensitivity'], results_df, 0, 0.04, match_number, self.modelType)
                thresholdOnTestSet = thresholds_on_test.mean()

                """ Perform federated evaluation on validation set (NB: If no validation set, we will use the test set) """
                predsOnValidationSet=siteTunedModel.predict_on_batch(X_val.values).ravel()
                loss = log_loss(Y_val,predsOnValidationSet)

                #Fully evaluate on the prospective set with 95% CIs
                #First, generated predicted labels based on selected threshold
                predictedLabels = np.where(predsOnValidationSet>thresholdOnTestSet,1,0)

                #Compare to global mdoel
                deLongPValue = utils.delong_roc_test(Y_val.values, global_model_predictions, predictedLabels)

                #Now fully evaluate at the threshold chosen using held out test set
                #Define the server round (if initial eval, set round to 0)
                recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, fpr, tpr = utils.calculateMetricsWithProbs (predictedLabels, predsOnValidationSet, Y_val, alpha)
                SiteSpecificModelFedEvalResults.loc[self.iteration,:] = [siteName, modelType, self.iteration, config['targetSensitivity'],imputationMethod,recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, deLongPValue, np.nan]

                #Encode the results in JSON, and transmit to central server
                #Make a dictionary of results, to transmit back to server
                siteSpecificModelResults = json.dumps(SiteSpecificModelFedEvalResults.loc[self.iteration,:].to_dict())

                return siteSpecificModelResults


        #Stage 4A: Logistic Regression: Perform Federated training using all of the training data for LR
        print ('## Model 1: Logistic Regression. Starting federated training - awaiting server.')
        modelType = "LR"
        model = LogisticRegression(
            penalty="l2",
            warm_start=True,  # prevent refreshing weights when fitting
        )

        #If model params are inherited from the 10CV exercise, can comment out line below to get params from the first-responding client
        utils.set_initial_params(model)
        originalServerAddress = serverAddress

        #Check code to offer opportunity to re-enter IP and port
        connected = False
        while not connected:
            try:
                print ("Federated Learning Server Address "+serverAddress)
                #Run Flower Client for LogisticRegression
                if sslEnabled==False:
                    LRFlowerClient = fl.client.start_numpy_client(server_address=serverAddress, client=LRClient())
                else:
                    LRFlowerClient = fl.client.start_numpy_client(server_address=serverAddress, client=LRClient(), root_certificates=Path("certificates/ca.crt").read_bytes())

                #If runs the client, connection has been successful
                connected = True

                print ("** Waiting "+str(pauseTime)+" seconds to allow DNN server to initialise")
                time.sleep(pauseTime)

                #Stage 4B: DNN Training
                print ('## Model 2: Deep Neural network.  Starting federated training - awaiting server.')

                modelType = "DNN"

                #Create instance of DNN model class (See Utils)
                model=utils.DNNModel(10)
                model.build((None,27))
                model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),metrics=['AUC'])

                #Run Flower Client for DNN
                if sslEnabled==False:
                    DNNClient = fl.client.start_numpy_client(server_address=serverAddress, client=DNNClient()) #root_certificates=Path("certificates/ca.crt").read_bytes()) #root_certificates=Path("certificates/ca.crt").read_bytes())
                else:
                    DNNClient = fl.client.start_numpy_client(server_address=serverAddress, client=DNNClient(), root_certificates=Path("certificates/ca.crt").read_bytes()) #root_certificates=Path("certificates/ca.crt").read_bytes())

            except Exception as e:
                print ('####################################################################')
                print ("### Error connecting to server ### Please contact the study lead")

                try:
                    hostName = utils.input_with_timeout("Enter IP address provided:",10)
                    if hostName == '':
                        serverAddress = originalServerAddress
                    else:
                            port = utils.input_with_timeout("Enter port provided:",20)
                            serverAddress = hostName+":"+port
                except Exception as e:
                    #If no IP address or port given, go back to the original values
                    serverAddress = originalServerAddress
                    continue;


        #Set file name based on the type of evaluation; if a training only site, this is a self evaluation. If there is a validation set, it is a prospective evaluation
        if not trainingOnly:
            #Write Federated Evaluation Results to file for global model & site specific fine-tuned model
            GlobalModelFedEvalResultsName = resultsPath + siteName + ' Global Model Federated Prospective Evaluation Results.csv'
            SiteSpecificModelFedEvalResultsName = resultsPath + siteName + ' Site-Tuned Federated Prospective Evaluation Results.csv'

        elif trainingOnly:
            #Write Federated Evaluation Results to file for global model & site specific fine-tuned model
            GlobalModelFedEvalResultsName = resultsPath + siteName + ' Federated Self Evaluation Results.csv'
            SiteSpecificModelFedEvalResultsName = resultsPath + siteName + ' Site-Tuned Federated Self Evaluation Results.csv'

        #Write out Federated Results
        GlobalModelFedEvalResults.to_csv(GlobalModelFedEvalResultsName)
        SiteSpecificModelFedEvalResults.to_csv(SiteSpecificModelFedEvalResultsName)

        print ('#######################################################################')
        print ('#######################################################################')
        print ('## Programme Completed Successfully - Thank you for your participation.')
        print ('## The CURIAL team is very grateful for your time and support')
        print ('## This programme will automatically re-run in 30 seconds')
        print ('## Please check with the study leads whether it would be helpful to leave the programme running')
        print ('## Once approved to disconnect you may delete the data from the device, and securely destroy the microSD card')
        print ('#######################################################################')
        time.sleep(30)
