#Script to pre-process BH data from source in to standardised format for FL

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',200)
from copy import deepcopy
import numpy as np

from copy import deepcopy
import random
import os
import time

def preprocess(pathToRawData):

    #Read in raw data
    try:
        dataset = pd.read_csv(pathToRawData)
    except FileNotFoundError:
        print("File "+pathToRawData+" not found.")
        time.sleep(10)
        exit()

    #Standardise column names to standard common names
    dataset = dataset.rename(columns={
        "PresentationID (unique identifier)":"ClusterID",
        "PresentationDate":"ArrivalDateTime",
        "AgeOnPresentation":"Age",
        "SEX (1 = FEMALE, 0 = MALE)":"Gender",
        "EthnicityCode":"Ethnicity",
        "AdmissionRespRate":"Vital_Sign Respiratory Rate",
        "AdmissionHeartRate":"Vital_Sign Heart Rate",
        "AdmissionBloodPressureSystolicBP":"Vital_Sign Systolic Blood Pressure",
        "AdmissionBloodPressureDiastolicBP":"Vital_Sign Diastolic Blood Pressure",
        "AdmissionSpO2":"Vital_Sign Oxygen Saturation",
        "AdmissionTemperature":"Vital_Sign Temperature Tympanic",
        "AdmissionOxygenDeliveryDevice":"Vital_Sign Delivery device used",
        "PresentationHAEMOGLOBIN":"Blood_Test HAEMOGLOBIN",
        "PresentationWHITE CELLS":"Blood_Test WHITE CELLS",
        "PresentationPLATELETS":"Blood_Test PLATELETS",
        "PresentationMEAN CELL VOL.":"Blood_Test MEAN CELL VOL.",
        "PresentationNEUTROPHILS":"Blood_Test NEUTROPHILS",
        "PresentationHAEMATOCRIT":"Blood_Test HAEMATOCRIT",
        "PresentationLYMPHOCYTES":"Blood_Test LYMPHOCYTES",
        "PresentationMONOCYTES":"Blood_Test MONOCYTES",
        "PresentationEOSINOPHILS":"Blood_Test EOSINOPHILS",
        "PresentationBASOPHILS":"Blood_Test BASOPHILS",
        "PresentationSODIUM":"Blood_Test SODIUM",
        "PresentationALBUMIN":"Blood_Test ALBUMIN",
        "PresentationALK.PHOSPHATASE":"Blood_Test ALK.PHOSPHATASE",
        "PresentationALT":"Blood_Test ALT",
        "PresentationUREA":	"Blood_Test UREA",
        "PresentationBILIRUBIN":"Blood_Test BILIRUBIN",
        "PresentationCREATININE":"Blood_Test CREATININE",
        "PresentationeGFR":"Blood_Test eGFR",
        "PresentationPOTASSIUM":"Blood_Test POTASSIUM",
        "PresentationCRP":"Blood_Test CRP",
        "PresentationAPTT":"Blood_Test APTT",
        "PresentationProthromb. Time":"Blood_Test Prothromb. Time",
        "PresentationPOCT pC02":"Blood_Gas pCO2 POC",
        "PresentationPOCT sO2":"Blood_Gas O2 Sat (BG)",
        "PresentationPOCT pO2":"Blood_Gas pO2 (BG)",
        "PresentationPCT cBASE(Ecf)c":"Blood_Gas BE Std (BG)",
        "PresentationPCT CO3(P,st)c":"Blood_Gas Bicarb (BG)",
        "PresentationPOCT Hctc":"Blood_Gas Hct (BG)",
        "PresentationPOCT cK+":"Blood_Gas K+ (BG)",
        "PresentationPOCT cNA+":"Blood_Gas Na+ (BG)",
        "PresentationPOCT cLAC":"Blood_Gas cLAC (BG)",
        "PresentationPOCT cCA++":"Blood_Gas Ca+ + (BG)",
        "PresentationPOCT ctHb":"Blood_Gas Hb (BG)",
        "PresentationGLUCOSE":"Blood_Gas Glucose (BG)"
    })

    #Process discharged location in to pre-determined admission outcome columns --- Outcome (0=discharged, 1=ward admission, 2=ICU admission, 3=death) --- same as bham
    outcome_dummy = pd.get_dummies(dataset['Outcome (0=discharged, 1=ward admission, 2=ICU admission, 3=death)'])
    outcome_dummy.columns = ["Discharged","Admission","ICU","death"]
    dataset = pd.concat([dataset,outcome_dummy],axis = 1)
    del dataset['Outcome (0=discharged, 1=ward admission, 2=ICU admission, 3=death)']

    #Standardise PCR results
    dataset['Covid-19 Positive'] = pd.to_numeric(dataset[['SARS-CoV-2 PCR','SAMBA Result (0 = INVALID, 1 = NEGATIVE, 2 = POSITIVE)']].max(axis=1))
    del dataset['SAMBA Result (0 = INVALID, 1 = NEGATIVE, 2 = POSITIVE)']
    del dataset['SARS-CoV-2 PCR']
    dataset['Covid-19 Positive'] = dataset['Covid-19 Positive'].replace({0.0:np.nan, 1.0:0.0, 2.0:1.0})

    #Standardise Oxygen values
    dataset['Vital_Sign Delivery device used'] = dataset['Vital_Sign Delivery device used'].replace(['NV','VENTURI','VM','NEBULISER', 'MASK','FACEMASK','NEB','FM','NIC','NCX','NC'],1)
    dataset['Vital_Sign Delivery device used'] = dataset['Vital_Sign Delivery device used'].replace(['NRB','NBM','NRBM','REBREATH','NRM','MRBM','NON REBREATH'],2)
    dataset['Vital_Sign Delivery device used'] = dataset['Vital_Sign Delivery device used'].replace(['CPAP','NIV','ETT','NIPPY','HIGH FLOW','H FLOW','NP','NV'],3)
    dataset['Vital_Sign Delivery device used'].fillna(0, inplace=True)
    dataset['Vital_Sign Delivery device used'] = dataset['Vital_Sign Delivery device used'].replace(['96'],np.nan)

    #Deal with extreme values
    dataset = dataset.replace({'<':''}, regex=True)
    dataset = dataset.replace({'>':''}, regex=True)
    #Standardise data format type
    dataset.loc[:,'Blood_Test HAEMOGLOBIN':'Blood_Gas Glucose (BG)'] = dataset.loc[:,'Blood_Test HAEMOGLOBIN':'Blood_Gas Glucose (BG)'].apply(pd.to_numeric, errors='coerce')

    #Standardise column datatype for Arrivals
    dataset['ArrivalDateTime'] = pd.to_datetime(dataset['ArrivalDateTime'], infer_datetime_format=True)

    #Remove invalids
    dataset = dataset[~(dataset['Covid-19 Positive'].isna())]

    #Standardise ethnicities to common format
    dataset['Ethnicity'] = ethnicityHarmoniser (dataset['Ethnicity'])

    #Save standardise form of dataset
    dataset.to_csv("../processedData/BH-CURIAL Processed Data.csv")

    #Print outcomes
    print('n patients:\t\t{}'.format(len(dataset.count(axis='columns'))))
    return dataset

#A function to harmonise ethnicities in to a standard form
def ethnicityHarmoniser(df):
        output = df.replace({
        "A": "White",
        "Z":"Unknown", "Z9":"Unknown",
        "C":"White", "C3":"White", "CP":"White", "CA":"White", "CB":"White", "CY":"White", "CC":"White", "CW":"White", "CK":"White","CS":"White", "CN":"White", "CR":"White", "CQ":"White", "CF":"White", "C2":"White", "CH":"White", "CU":"White", "CHI":"White",
        "J":"South Asian",
        "L":"Other", "LK":"Other", "LJ":"Other", "LA":"Other", "LH":"Other", "LE":"Other", "LD":"Other", "LG":"Other",
        "H":"South Asian",
        "B":"White",
        "S":"Other", "SE":"Other", "SC":"Other", "SD":"Other", "SA":"Other",
        "N":"Black", "NK":"Black",
        "M":"Black",
        "G":"Mixed", "GF":"Mixed", "GB":"Mixed",
        "P":"Black", "PE":"Black", "PD":"Black", "PC":"Black", "PA":"Black",
        "K":"South Asian",
        "D":"Mixed", "CX":"Mixed", "GE":"Mixed", "GD":"Mixed", "GA":"Mixed",
        "R":"Chinese",
        "F":"Mixed",
        "E":"Mixed",
        "Z":"Unknown","ZR":"Unknown",
        "White - British":"White",
        "White- British":"White",
        "Other- Not stated":"Unknown",
        "White- Any other white background":"White",
        "Other - Not Stated":"Unknown",
        "White- Irish":"White",
        "Asian or Asian British - Pakistani":"South Asian",
        "Other- Any other ethnic group":"Other",
        "Black- Any other black background":"Black",
        "White - Any Other White Background":"White",
        "Asian or Asian British - Indian":"South Asian",
        "Asian - Any Other Asian Background":"Other",
        "Other - Not Known":"Unknown",
        "Black or Black British - Caribbean":"Black",
        "Other- Not known":"Unknown",
        "Asian or Asian British -Indian":"South Asian",
        "Other - Any Other Ethnic Group":"Other",
        "Asian or Asian British - Bangladeshi":"South Asian",
        "Mixed - White and Black African":"Mixed",
        "Mixed - White and Black Caribbean":"Mixed",
        "Asian or Asian British -Any other Asian background":"Other",
        "Other-Chinese":"Chinese",
        "1":"Unknown", "WHT":"Unknown","62":"Unknown","19":"Unknown","87":"Unknown",
        })
        return output
