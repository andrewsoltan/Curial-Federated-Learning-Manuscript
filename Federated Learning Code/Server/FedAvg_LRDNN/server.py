##################################################################
#
#   CURIAL-Federated: A Federated Learning Pipeline for COVID-19 screening
#   Server Pipeline
#   Version: v1.0
#
##################################################################

import sys
import warnings
import flwr as fl
from sklearn.metrics import log_loss
from typing import Dict
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import openpyxl
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, recall_score, precision_score, accuracy_score,roc_auc_score, confusion_matrix, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.special import ndtri
import math
import pickle
import json

import tensorflow as tf
#Switch to v1 behaviour; required for backwards compatiability with SHAP library
#tf.compat.v1.disable_v2_behavior()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model

import random
import os
import time

import utils

UHB=0
OUH=0

##############################################
# FEDERATED LEARNING SERVER
##############################################

##############################################
# Global configuration: FL server parameters #
# Confirm parameters prior to deployment #
##############################################
#hostName = "10.0.0.5"
hostName = "localhost"
sslEnabled = True
port = "8002"
serverAddress = hostName+":"+port
n_rounds_LR = 15 #Set how many rounds of FL to perform
n_rounds_DNN = 15 #Set how many rounds of FL to perform

min_fit_clients = 3 #Set minimum number of contributing clients to run FL exercise
min_available_clients = 4 #Set minimum number of required clients for the scripts to run (=n sites)

#Training parameters
LRmaxIter = 5
nepochs = 50 #n epochs during DNN training
earlystoppingPatience = 15 #n steps without improvement prior to stopping
batchSize=4096

##############################################
# Configuration for evaluation script #
##############################################
global siteName
siteName = "BH"
scaleUsing = "OUH"
evaluationOnly = True
pathToValidationData = "../raw/Curial Data v.1.csv"

#Import pre_processing for site-specific dataset from raw file
sys.path.insert(0, '../preproc')
import preprocess_BH as preprocessor

##################################
# Validation analysis constants #
##################################
featureSet = "Bloods & Vitals" #, 'OLO & Vitals', 'Bloods & Blood_Gas & Vitals'];
imputationMethod = "median"
localpath = ''
resultsPath = 'Results/'

##############################
# Deep learning parameters #
##############################
tf.keras.backend.set_floatx('float32')

##############################
# Evaluation Metrics #
##############################
metric_names = ['Recall','Precision','F1-Score','Accuracy','Specificity','PPV','NPV','AUC', 'DeLong P Current vs Global', 'DeLong LR vs DNN']
metrics = [recall_score,precision_score,accuracy_score,roc_auc_score]#,confusion_matrix]
metrics_dict = dict(zip(metric_names,metrics))
alpha = 0.95
targetSensitivity = 0.85

global meanCrossSiteThreshold
global meanOfMediansJson
meanCrossSiteThreshold=0
meanOfMediansJson = ''

""" Import crucial data processing functions for server side evaluation """
""" Generate X & Y for testing  """
def load_data_splits(df,featureList):
        X = df[featureList]
        Y = df['Covid-19 Positive']
        return X,Y

def LRfit_config(server_round: int) -> Dict:
    """Send round number to client."""

    config = {
        "max_iterations": LRmaxIter,
        "server_round": server_round
    }
    return config

def DNNfit_config(server_round: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": batchSize,
        "local_epochs": nepochs,
        "patience": earlystoppingPatience,
        "server_round": server_round
    }
    return config

def LRevaluate_config(server_round: int) -> Dict:
    """Send mean threshold & imputation values to client."""
    return {"meanCrossSiteThreshold": meanCrossSiteThreshold, "meanOfMedians":meanOfMediansJson, "server_round": server_round, "targetSensitivity":targetSensitivity}

def DNNevaluate_config(server_round: int):
    """Return training configuration dict for each round.
    """
    config = {
        "batch_size": batchSize,
        "local_epochs": nepochs,
        "patience": earlystoppingPatience,
        "meanCrossSiteThreshold": meanCrossSiteThreshold,
        "meanOfMedians":meanOfMediansJson,
        "server_round": server_round,
        "targetSensitivity": targetSensitivity
    }
    return config

#Implement custom strategy aggregator to allow threshold transmission (see https://flower.dev/docs/saving-progress.html)
class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    GlobalModelResults_dict = {}
    SiteSpecificModelresults_dict = {}
    TestSetresults_dict = {}
    ROCanalysis_df = pd.DataFrame()

    i=1
    j=1
    k=1

    def aggregate_fit(
        self,
        server_round: int,
        results: List,
        failures: List,
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        global meanOfMediansJson

        """ Process Median values and perform averaging """
        medians = [r.metrics["trainingPopulationMedian"] for _, r in results]
        allmedians = pd.DataFrame()
        #Iterate through medians, putting in to a df
        for median in medians:
            temp = pd.Series(json.loads(median))
            allmedians = pd.concat([allmedians,temp],axis=1)
        #Calculate mean of medians
        meanOfMedians = allmedians.mean(axis=1)

        #Convert to JSON
        meanOfMediansJson = json.dumps(meanOfMedians.to_dict())

        # """ Aggregate SHAP values """
        # shap_values = [r.metrics["shap_values"] for _, r in results]
        # allSHAP = pd.DataFrame()
        # #Iterate through shap values, putting in to a df
        # for shaps in shap_values:
        #     temp = pd.Series(json.loads(shaps))
        #     allmedians = pd.concat([shap_values,temp],axis=1)

        return super().aggregate_fit(server_round, results, failures)

    def rocCurveResultsTable (self, results_dict, resultsPath, modelType, fileName):
        resultsDf = pd.DataFrame.from_dict(results_dict)
        #Group by site ID
        groups = resultsDf.groupby(["Site ID"])
        sites = list(groups.groups.keys())
        metrics = ['TPR','FPR'];

        #Create dataframe with multi-level index of sites & metrics
        df = pd.DataFrame(columns=pd.MultiIndex.from_product([sites,metrics]))
        df = df.shift()[1:]
        for group in groups.groups.keys():
            df[group] = groups.get_group(group).reset_index()[['TPR','FPR']]
        df.index.rename('Value', inplace=True)
        #print(df)

        filePath = resultsPath + modelType + ' ' + fileName + ' - Results collated on server.csv'

        return df, filePath

    def generateResultsTable (self, results_dict, resultsPath, modelType, fileLabel):
        resultsDf = pd.DataFrame.from_dict(results_dict)
        #Transpose Results to give Vertical File
        transpose = resultsDf.transpose()

        #Group by site ID
        groups = transpose.groupby(["Site ID"])

        sites = list(groups.groups.keys())
        metrics = ['AUROC','Achieved Sensitivity', 'Specificity', 'Accuracy','Precision','NPV','F1','n False Negative','n False Positive','DeLong P Current vs Global', 'DeLong LR vs DNN'];

        #Create dataframe with multi-level index of sites & metrics
        df = pd.DataFrame(columns=pd.MultiIndex.from_product([sites,metrics]))
        df = df.shift()[1:]
        for group in groups.groups.keys():
            df[group] = groups.get_group(group).reset_index()[['AUROC','Achieved Sensitivity', 'Specificity', 'Accuracy','Precision','NPV','F1','n False Negative','n False Positive', 'DeLong P Current vs Global', 'DeLong LR vs DNN']]
        df.index.rename('Iteration', inplace=True)
        filePath = resultsPath + modelType + ' ' + fileLabel + ' - Results collated on server.csv'

        return df, filePath

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List,
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None
        global meanCrossSiteThreshold

        """ Find mean threshold to send for Federated & Centralised evaluation"""
        thresholds = [r.metrics["Threshold"] for _, r in results]
        #Remove NA values coming from sites that are evaluating only
        #print(thresholds)
        thresholds = [x for x in thresholds if np.isnan(x) == False]
        #print(thresholds)
        meanCrossSiteThreshold = sum(thresholds)/len(thresholds)
        print ("Thresholds mean " + str(meanCrossSiteThreshold))

        """ Extract and save values to plot ROC curves on Final Iteration """
        roc_analysis_iter = [r.metrics["ROC_analysis"] for _, r in results]
        for roc_set in roc_analysis_iter:
                result = json.loads(roc_set)
                rocAnalysis, rocAnalysisPath = self.rocCurveResultsTable(result, resultsPath, modelType, 'ROC analysis')
                self.ROCanalysis_df = pd.concat([self.ROCanalysis_df,rocAnalysis], axis=1)

        #For each round, save ROC analysis
        self.ROCanalysis_df.to_csv(rocAnalysisPath)

        #The first round is a special case as the ROC curves are for the index models; save the first-round ROC curves separately
        if server_round == 1:
            filePath = resultsPath + modelType + ' First Round ROC curves - Results collated on server.csv'
            self.ROCanalysis_df.to_csv(filePath)

        #Clean ROC analysis after each iter to get last iterations
        self.ROCanalysis_df = pd.DataFrame()

        """ Extract & save test set results from client messages """
        TestSetIterResults = [r.metrics["Test_Set_Iter_Results"] for _, r in results]
        #Remove NA values coming from sites that are evaluating only
        TestSetIterResults = [x for x in TestSetIterResults if pd.notnull(x)]
        for IterationResult in TestSetIterResults:
            #Decode the JSON, and re-encode as dataframe
            result = json.loads(IterationResult)
            self.TestSetresults_dict[self.k] = result
            self.k = self.k+1
        #Generate formatted results table for tthe Test Set
        TestSetIterResults, TestSetIterResultsPath = self.generateResultsTable(self.TestSetresults_dict, resultsPath, modelType, 'Test Set Evaluation')
        TestSetIterResults.to_csv(TestSetIterResultsPath)


        """ Global Model Fed Eval: Extract the Evaluation Results from client messages for the global model """
        GlobalModelIterationResults = [r.metrics["Global_Model_Iter_Results"] for _, r in results]
        for IterationResult in GlobalModelIterationResults:
            #Decode the JSON, and re-encode as dataframe
            result = json.loads(IterationResult)
            self.GlobalModelResults_dict[self.j] = result
            self.j = self.j+1

        #Generate formatted results table for the Global Model
        globalModelFedEvalResults, globalModelFederatedEvalResultsPath = self.generateResultsTable(self.GlobalModelResults_dict, resultsPath, modelType, 'Global Model Federated Evaluation')
        globalModelFedEvalResults.to_csv(globalModelFederatedEvalResultsPath)
        print(globalModelFedEvalResults)

        """ Site-Personalised Model Fed Eval: Extract the Evaluation Results from client messages for the Site-Personalised model """
        SitePersonalisedModelIterationResults = [r.metrics["Site_Personalised_Iter_Results"] for _, r in results]
        #Remove NA values coming from sites that are evaluating
        SitePersonalisedModelIterationResults = [x for x in SitePersonalisedModelIterationResults if pd.notnull(x)]
        for IterationResult in SitePersonalisedModelIterationResults:
            #Decode the JSON, and re-encode as dataframe
            result = json.loads(IterationResult)
            self.SiteSpecificModelresults_dict[self.i] = result
            self.i = self.i+1

        #Generate formatted results table for the Global Model
        sitePersonalisedModelFedEvalResults, sitePersonalisedModelFederatedEvalResultsPath = self.generateResultsTable(self.SiteSpecificModelresults_dict, resultsPath, modelType, 'Site Personalised Federated Evaluation')
        sitePersonalisedModelFedEvalResults.to_csv(sitePersonalisedModelFederatedEvalResultsPath)

        return super().aggregate_evaluate(server_round, results, failures)


# Call aggregate_evaluate from base class (FedAvg)
#return super().aggregate_evaluate(server_round, results, failures)
""" Perform server side evaluation on the Validation Dataset (BH in server file) """
def get_evaluate_fn(model):
    #Set 1: Load server side validation set
    serverSideValidationSet = preprocessor.preprocess(pathToValidationData)
    #Restrict Population to patients admitted to Hospital
    serverSideValidationSet = serverSideValidationSet [(serverSideValidationSet.Admission == 1.0) | (serverSideValidationSet.ICU == 1.0)]

    #Load Scaler pickle files for imputing and scaling - apply to dataset/transform
    scalarFile = localpath+'norm_'+scaleUsing+'.pkl'
    scaler = pd.read_pickle(scalarFile)
    features = utils.featureList(featureSet)

    #Load data splits
    X,Y_val = load_data_splits(serverSideValidationSet, features)

    # # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):

        global X_val
        global rocAnalysisFederatedEval

        #Skip round if not yet received mean of medians
        if meanOfMediansJson == '':
            return None

        #Impute missing values based on mean of medians
        meanOfMedians = json.loads(meanOfMediansJson)
        X_val = X.fillna(value=meanOfMedians)

        #Perform scaling
        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

        if modelType == "LR":
            utils.set_model_params(model, parameters)
            preds=model.predict_proba(X_val)[:,1]
        elif modelType == "DNN":
            model.set_weights(parameters)
            preds = model.predict_on_batch(X_val).ravel()
            server_round = server_round + n_rounds_LR #Increase server round raw number for DNN to avoid overwriting results

        loss = log_loss(Y_val,preds)
        auc_95CI=utils.AUC_CIs (preds, Y_val, alpha = 0.95)
        print ("Round " + str(server_round) + ", AUROC: " + auc_95CI)

        #Fully evaluate on the prospective set with 95% CIs
        #First, generated predicted labels based on selected threshold
        predictedLabels = np.where(preds>meanCrossSiteThreshold,1,0)

        """Generate ROC curve on server-side evaluation set """
        testset_fpr, testset_tpr, _ = roc_curve(Y_val, preds)
        rocAnalysisFederatedEval = pd.concat([pd.Series(testset_fpr),pd.Series(testset_tpr)], axis=1)
        rocAnalysisFederatedEval = rocAnalysisFederatedEval.rename(columns= {0: "FPR", 1: "TPR"})

        recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score, fpr, tpr = utils.calculateMetricsWithProbs (predictedLabels, preds, Y_val, alpha)
        serverSideEval.loc[server_round,:] = [siteName, modelType, targetSensitivity,imputationMethod,recallAchieved,specificity,accuracy,auc,precision,npv,fn,fp,f1score]

        return loss,  {"AUROC": auc_95CI}
    return evaluate


# Start Flower server for 20 rounds of federated learning
if __name__ == "__main__":
    #Create Ouptut Dataframe for Server Side Evaluation Matrix
    serverSideEval = pd.DataFrame(columns=['Site Name','Model Type','CURIAL Targetted Recall','Imputation','Achieved Sensitivity','Specificity','Accuracy','AUROC','Precision','NPV','n False Negative','n False Positive','F1'])

    #First create instance of LR model and test
    model = LogisticRegression(
        penalty="l2",
        max_iter=LRmaxIter,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    modelType = "LR"
    utils.set_initial_params(model)

    #Create instance of custom aggregated strategy
    LRstrategy = AggregateCustomMetricStrategy(
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=LRfit_config,
        on_evaluate_config_fn=LRevaluate_config,
        evaluate_fn=get_evaluate_fn(model)
    )

    #Run FL Side Server
    if sslEnabled == False:
        LR=fl.server.start_server(
            server_address=serverAddress,
            strategy=LRstrategy,
            config=fl.server.ServerConfig(num_rounds=n_rounds_LR),
        )
    else:
            LR=fl.server.start_server(
                server_address=serverAddress,
                strategy=LRstrategy,
                config=fl.server.ServerConfig(num_rounds=n_rounds_LR),
                certificates=(Path("certificates/ca.crt").read_bytes(),Path("certificates/server.pem").read_bytes(),Path("certificates/server.key").read_bytes())
            )

    #Model File Name
    name = 'Results/SavedModel_' + modelType + ' Global Model.pkl'

    #Save Global LR model
    with open(os.path.join(localpath, name),'wb') as f:
        pickle.dump(model,f)

    """ Save LR Model for SHAP analysis """
    LRmodel = model

    #Save ROC analysis for Centralised Evaluation
    name = 'Results/ROCanalysis_CentralisedEval' + modelType + ' ' + siteName + ' Global Model.pkl'
    rocAnalysisFederatedEval.to_csv(name)

    print('========================== Model 1 Complete. Average threshold across all sites ' + str(meanCrossSiteThreshold))

    print('========================== Model 2: DNN - Running server')

    #Create instance of DNN model class (See Utils)
    modelType = "DNN"
    model = utils.DNNModel(10)
    model.build((None,27))
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),metrics=['AUC'])

    # Create strategy
    DNNstrategy = AggregateCustomMetricStrategy(
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=DNNfit_config,
        on_evaluate_config_fn = DNNevaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    #Clean results dict
    DNNstrategy.TestSetresults_dict = {}
    DNNstrategy.GlobalModelResults_dict = {}
    DNNstrategy.SiteSpecificModelresults_dict = {}

    #Run Flower server for DNN
    if sslEnabled = True;
        DNN=fl.server.start_server(
            server_address=serverAddress,
            config=fl.server.ServerConfig(num_rounds=n_rounds_DNN),
            strategy=DNNstrategy
        )
    else:
        DNN=fl.server.start_server(
            server_address=serverAddress,
            config=fl.server.ServerConfig(num_rounds=n_rounds_DNN),
            strategy=DNNstrategy
            certificates=(Path("certificates/ca.crt").read_bytes(),Path("certificates/server.pem").read_bytes(),Path("certificates/server.key").read_bytes())
        )

    ####### If re-enabling SHAP: remember to enable v1 backwards compatability
    # #Perform SHAP analysis for final LR and DNN models after 150 iters
    # """ Perform Centralised SHAP evaluation ON FINAL MODEL using BH set for positives """
    # ind=np.where(Y_val==1)[0]
    # shapExplainer = shap.DeepExplainer(model, X_val.iloc[ind])
    # shapValues = shapExplainer.shap_values(X_val.iloc[ind].values)
    # name = 'Results/shapExplanations_DNN_final_global_model.pkl'
    # #Save explainer item
    # with open(name, "wb") as f:
    #     pickle.dump(shapValues,f)

    """ LR: Perform Centralised SHAP evaluation ON FINAL LR MODEL using BH set for positives """
    print ("#### Performing SHAP analysis on LR Model")
    explainer = shap.Explainer(LRmodel.predict, X_val)
    shapValues = explainer(X_val)
    name = 'Results/shapExplanations_LR_final_global_model.pkl'
    #Save explainer item
    with open(name, "wb") as f:
        pickle.dump(shapValues,f)

    """ DNN: Perform Centralised SHAP evaluation ON FINAL LR MODEL using BH set for positives """
    print ("#### Performing SHAP analysis on DNN Model")
    explainer = shap.Explainer(model.predict, X_val.sample(1000))
    shapValues = explainer(X_val.sample(400))
    name = 'Results/shapExplanations_DNN_final_global_model.pkl'
    with open(name, "wb") as f:
         pickle.dump(shapValues,f)

    #Model File Name
    name = 'Results/SavedModel_' + modelType + ' Global Model.pkl'
    weightsFileName = 'Results/SavedModelWeights_' + modelType + ' Global Model.h5'

    #Save Global DNN model
    with open(os.path.join(localpath, name),'wb') as f:
        pickle.dump(model,f)

    #Dump model weights in to a separate file (to allow tensorflow loading)
    model.save_weights(weightsFileName)

    #Save ROC analysis for Centralised Evaluation
    name = 'Results/ROCanalysis_CentralisedEval' + modelType + ' ' + siteName + ' Global Model.pkl'
    rocAnalysisFederatedEval.to_csv(name)

    #Write Server Side Eval Results to File
    serverSideEvalName = resultsPath + siteName + ' Server Prospective Evaluation Results.csv'

    #Write out Federated Results
    serverSideEval.to_csv(serverSideEvalName)
