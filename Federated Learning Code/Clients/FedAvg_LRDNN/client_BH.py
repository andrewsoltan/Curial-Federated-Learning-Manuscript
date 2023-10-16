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
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, recall_score, precision_score, accuracy_score,roc_auc_score, confusion_matrix, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
sslEnabled=True
#hostName = "localhost"
port = "8002"
pauseTime = 10 #Set time pause to wait after completing LR and running DNN
serverAddress = hostName+":"+port
relaunchOnCompletion = True

##############################################
# SITE-SPECIFIC CONFIGURATION #
##############################################
global siteName
siteName = "BHEvaluation"
evaluationOnly = True
pathToValidationData = "../raw/Curial Data v.1.csv"
validationFromDate = '2021-01-01'

#Import pre_processing for site-specific dataset from raw file
sys.path.insert(1, '../preproc')
import preprocess_BH as preprocessor

##################################
# Validation analysis constants #
##################################
featureSet = "Bloods & Vitals" #, 'OLO & Vitals', 'Bloods & Blood_Gas & Vitals'];
imputationMethod = "median"
localpath = ''
resultsPath = 'Results/'
scaleUsing = "OUH"

#Import pre_processing for site-specific dataset from raw file
#Set present working directory to the true file location (to permit shortcut launching on Linux)
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, '../preproc')
sys.path.insert(2, '..')
import preprocess_BH as preprocessor

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

    while relaunchOnCompletion == True:
        ##############
        #Stage 1: Run pre-processing script from raw dataset
        print ('Importing and pre-processing dataset')
        #First pre-process dataset
        fullDataset = preprocessor.preprocess(pathToValidationData)

        ##############
        #Stage 2: Formation of validation cohorts (where applicable), noramlisation & imputation
        #Restrict Population to patients admitted to Hospital
        fullDataset = fullDataset[(fullDataset.Admission == 1.0) | (fullDataset.ICU == 1.0)]

        #Enforce a validation from date
        validationDataset = fullDataset[fullDataset.ArrivalDateTime >= validationFromDate]

        #Restrict validation to patients who had a Covid-19 test result
        validationDataset = validationDataset[(validationDataset['Covid-19 Positive'] == 1.0) | (validationDataset['Covid-19 Positive'] == 0.0)]

        print ("Validation set: Covid-19 Cases " + str(validationDataset['Covid-19 Positive'].sum()))
        print ("Validation set: Total length " + str(validationDataset.shape[0]))

        """ Generate X & Y for training  """
        def load_data_splits(df,featureList):
                X = df[featureList]
                Y = df['Covid-19 Positive']
                return X,Y

        #Get Features, generate X & Y
        features = utils.featureList(featureSet)
        X,Y_val = load_data_splits(validationDataset, features)

        #Load Imputer, Scaler pickle files for imputing and scaling (NB: Here we are using OUH median & scaler) - apply to dataset/transform
        #imputerFile =  localpath+'imputer_'+scaleUsing+'_'+imputationMethod+'.pkl'
        scalarFile = localpath+'norm_'+scaleUsing+'.pkl'
        #imputer = pd.read_pickle(imputerFile)
        scaler = pd.read_pickle(scalarFile)

        #X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

        #Create Ouptut Dataframe for Prospective Evaluation Matrix
        GlobalModelFedEvalResults = pd.DataFrame(columns=['Site ID','Model Type', 'Iteration', 'CURIAL Targetted Recall','Imputation','Achieved Sensitivity','Specificity','Accuracy','AUROC','Precision','NPV','n False Negative','n False Positive','F1','DeLong P Current vs Global','DeLong LR vs DNN'])
        LRValidationSetPreds = np.zeros(1)

        ##############
        """ Define Logistic Regression Federated Flower Client & evaluation"""
        class LRClient(fl.client.NumPyClient):
            iteration = 0
            modelType = "LR"
            global siteName

            def get_parameters(self, config):  # type: ignore
                return utils.get_model_parameters(model)

            #Stage 3: Federated evaluation on prospective/true validation set
            """ Perform Training, Calibration on the Test set, assessment on the Test Set, followed by Federated Evaluation of the global model on the validation set """
            def evaluate(self, parameters, config):

                print(f"Round {config['server_round']}")

                #If the client successfully runs, store the last successful IP as the new default IP
                originalServerAddress = serverAddress

                #Get parameters of Global Model
                utils.set_model_params(model, parameters)
                self.iteration = self.iteration + 1

                #Skip round if not yet received mean of medians
                if config['meanOfMedians'] == '':
                    return None

                #Impute missing values based on mean of medians
                meanOfMedians = json.loads(config['meanOfMedians'])

                #Impute using mean of medians
                X_val = X.fillna(value=meanOfMedians)

                #Scaling
                X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

                """ Determine optimal threshold from config / mean of other participating sites"""
                federatedThreshold = config['meanCrossSiteThreshold']

                """ Perform federated evaluation on validation set (NB: If no validation set, we will use the test set) """
                predsOnValidationSet=model.predict_proba(X_val)[:,1]
                loss = log_loss(Y_val,predsOnValidationSet)

                #Fully evaluate on the prospective set with 95% CIs
                #First, generated predicted labels based on selected threshold set during testing
                predictedLabels = np.where(predsOnValidationSet>federatedThreshold,1,0)
                #Now fully evaluate at the threshold chosen using held out test set
                #Define the server round (if initial eval, set round to 0)
                recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, fpr, tpr = utils.calculateMetricsWithProbs (predictedLabels, predsOnValidationSet, Y_val, alpha)
                GlobalModelFedEvalResults.loc[self.iteration,:] = [siteName, modelType, self.iteration, config['targetSensitivity'],imputationMethod,recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, np.nan, np.nan]
                print (auc)
                #Encode the federated evaluation results in JSON, and transmit to central server
                GlobalModelResults_dict = json.dumps(GlobalModelFedEvalResults.loc[self.iteration,:].to_dict())

                """Set a global variable with the LR Validation Set Preds to allow later adjustment"""
                global LRValidationSetPreds
                LRValidationSetPreds = predictedLabels

                """ Calculate variables for a VALIDATION ROC curve """
                valset_fpr, valset_tpr, _ = roc_curve(Y_val, predsOnValidationSet)
                rocAnalysisOnFinalIter = pd.concat([pd.Series(valset_fpr),pd.Series(valset_tpr)], axis=1)
                rocAnalysisOnFinalIter = rocAnalysisOnFinalIter.rename(columns= {0: "FPR", 1: "TPR"})
                rocAnalysisOnFinalIter["Site ID"] = siteName
                rocAnalysisDict = json.dumps(rocAnalysisOnFinalIter.to_dict())

                return loss, len(X_val), {"Threshold":np.nan, "AUROC": auc, "ROC_analysis": rocAnalysisDict, "Global_Model_Iter_Results": GlobalModelResults_dict, "Test_Set_Iter_Results": np.nan, "Site_Personalised_Iter_Results": np.nan}


        ##############
        #Stage 4B: Deep Neural network - define Federated Flower Client
        class DNNClient(fl.client.NumPyClient):
            modelType = "DNN"
            iteration = 0
            global siteName

            #On intialisation of the class, set an offset to the number of iterations to keep the tracking on output variable
            def __init__(self):
                self.iteration = self.iteration + GlobalModelFedEvalResults.shape[0]

            """ Evaluate Global Model on Prospective Validation Set """
            def evaluate(self, parameters, config):
                print(f"Round {config['server_round']}")

                """" Update model with global parameters """
                model.set_weights(parameters)
                self.iteration = self.iteration + 1

                """ Skip Round if not yet got mean of medians """
                if config['meanOfMedians'] == '':
                    return None

                """ Impute missing values based on the mean of Medians, obtained from the configuration via federated setup"""
                meanOfMedians = json.loads(config['meanOfMedians'])
                X_val = X.fillna(value=meanOfMedians)

                """Perform scaling after imputation"""
                X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

                """ Determine optimal threshold from config / mean of other participating sites"""
                federatedThreshold = config['meanCrossSiteThreshold']

                """ Perform federated evaluation on validation set (NB: If no validation set, we will use the test set) """
                predsOnValidationSet=model.predict_on_batch(X_val).ravel()
                loss = log_loss(Y_val,predsOnValidationSet)

                #Fully evaluate on the prospective set with 95% CIs
                #First, generated predicted labels based on selected threshold
                predictedLabels = np.where(predsOnValidationSet>federatedThreshold,1,0)

                """ Perform DeLong's Testing to compare current DNN model with the saved global LR predictions """
                global LRValidationSetPreds
                LRvsDNNDeLongP = utils.delong_roc_test(Y_val.values, LRValidationSetPreds, predictedLabels)

                #Now fully evaluate at the threshold chosen using held out test set
                #Define the server round (if initial eval, set round to 0)
                recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, fpr, tpr = utils.calculateMetricsWithProbs (predictedLabels, predsOnValidationSet, Y_val, alpha)
                GlobalModelFedEvalResults.loc[self.iteration,:] = [siteName, modelType, self.iteration, config['targetSensitivity'],imputationMethod,recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, np.nan, LRvsDNNDeLongP]
                print (auc)
                #Encode the results in JSON, and transmit to central server
                #Make a dictionary of results, to transmit back to server
                GlobalModelResults_dict = json.dumps(GlobalModelFedEvalResults.loc[self.iteration,:].to_dict())

                """ Calculate variables for a ROC curve -- transmit to server """
                valset_fpr, valset_tpr, _ = roc_curve(Y_val, predsOnValidationSet)
                rocAnalysisOnFinalIter = pd.concat([pd.Series(valset_fpr),pd.Series(valset_tpr)], axis=1)
                rocAnalysisOnFinalIter = rocAnalysisOnFinalIter.rename(columns= {0: "FPR", 1: "TPR"})
                rocAnalysisOnFinalIter["Site ID"] = siteName
                rocAnalysisDict = json.dumps(rocAnalysisOnFinalIter.to_dict())

                #Message results back to server
                return loss, len(X_val), {"Threshold":np.nan, "AUROC": auc, "ROC_analysis": rocAnalysisDict, "Global_Model_Iter_Results": GlobalModelResults_dict, "Test_Set_Iter_Results": np.nan,  "Site_Personalised_Iter_Results": np.nan}

        #Stage 4A: Logistic Regression: Perform Federated training using all of the training data for LR
        print ('## Model 1: Logistic Regression. Starting federated evaluation - awaiting server.')
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
        if evaluationOnly:
            #Write Federated Evaluation Results to file for global model & site specific fine-tuned model
            GlobalModelFedEvalResultsName = resultsPath + siteName + ' Global Model Federated Prospective Evaluation Results.csv'

        #Write out Federated Results
        GlobalModelFedEvalResults.to_csv(GlobalModelFedEvalResultsName)
        print ('#######################################################################')
        print ('#######################################################################')
        print ('## Programme Completed Successfully - Thank you for your participation.')
        print ('## The CURIAL team is very grateful for your time and support')
        print ('## This programme will automatically re-run in 30 seconds')
        print ('## Please check with the study leads whether it would be helpful to leave the programme running')
        print ('## Once approved to disconnect you may delete the data from the device, and securely destroy the microSD card')
        print ('#######################################################################')
        time.sleep(30)
