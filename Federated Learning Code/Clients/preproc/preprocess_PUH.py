#Script to pre-process PUH data from source in to standardised format for FL

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

def preprocess(fileName, PCR48hWindowFile=False, era="Pandemic", siteName="PUH"):

    #Load Dataset
    try:
        df_raw = pd.read_csv(fileName)
    except FileNotFoundError:
        print("File "+fileName+" not found.")
        time.sleep(10)
        quit()

    #Standardise to a common date format across all sites
    df2 = df_raw[["YY", "MM"]].copy()
    df2.columns = ["year", "month"]
    df2['day'] = 1
    df_raw['ArrivalDateTime'] = pd.to_datetime(df2[["year", "month", "day"]])

    #Standardise vital signs across sites by merging where a hospital uses two different systems to collate
    df_pt = df_raw
    df_pt['ED sys_bp'].fillna(df_pt['sys_bp'], inplace=True)
    df_pt['ED dia_bp'].fillna(df_pt['dia_bp'], inplace=True)
    df_pt['ED Pulse'].fillna(df_pt['pulse'], inplace=True)
    df_pt['ED resp_rate'].fillna(df_pt['resp_rate'], inplace=True)
    df_pt['ED sats'].fillna(df_pt['sats'], inplace=True)
    df_pt['ED Temp'].fillna(df_pt['TempR'], inplace=True)
    oxygen_mask_ids = {0: 'L0', 1: 'L1', 2: 'L1', 3:'L2', 4: 'L1', 5:'L3', 6:'L3', 7:'L3', 8:'L2', 9:'L3', 10:'L3', 14:'L3',
                     16:'L1', 17:'L3', 18:'L3', 19:'L3', 20:'L1',21:'L3',22:'L3',23:'L3',27:'L3', 28:'L1', 29:'L3'}
    df_pt['oxygen_mask'].replace(oxygen_mask_ids, inplace=True)

    #Standardise Oxygen Level/Flowrates to common format: 0 = Room air; 1 =  1-9L/min AND 24%-35%; 2 = 10-20L, 40%; 3 = > 20L/min, 60%, 100%
    ed_o2_flowrate_ids = {'Room Air':'L0', '1 l/min':'L1', '1 l/Min':'L1', '2 l/min':'L1', '3 l/min':'L1', '4 l/min':'L1', '5 l/min':'L1',
                         '6 l/min':'L1','7 l/min':'L1','8 l/min':'L1','9 l/min':'L1', '24%':'L1', '28%':'L1','35%':'L1',
                         '10 l/min':'L2', '11 l/min':'L2','12 l/min':'L2','13 l/min':'L2','14 l/min':'L2','15 l/min':'L2',
                         '40%':'L2', '50%':'L3', '80%':'L3', '60%':'L3', '100%':'L3', '0': 'L0', '1': 'L1', '2': 'L1', '3':'L2', '4': 'L1', '5':'L3', '6':'L3', '7':'L3', '8':'L2', '9':'L3', '10':'L3', '14':'L3', '16':'L1', '17':'L3', '18':'L3', '19':'L3', '20':'L1','21':'L3','22':'L3','23':'L3','27':'L3', '28':'L1', '29':'L3'}
    df_pt['ED O2 Flowrate'].replace(ed_o2_flowrate_ids, inplace=True)
    df_pt['ED O2 Flowrate'].fillna(df_pt['oxygen_mask'], inplace=True)
    df_pt.drop(['oxygen_mask'], axis=1, inplace=True)
    df_pt.rename(columns={'ED O2 Flowrate':'Vital_Sign Delivery device used'}, inplace=True)

    #Standardise common names to the common format
    df_pt.rename(columns={'epis_sg2':'ClusterID'}, inplace=True)
    df_pt.rename(columns={'adm_age':'Age', 'gender':'Gender', 'ethnic_origin':'Ethnicity'}, inplace=True)
    df_pt.rename(columns={'ED Pulse':'Vital_Sign Heart Rate', 'ED resp_rate':'Vital_Sign Respiratory Rate',
                                  'ED sys_bp':'Vital_Sign Systolic Blood Pressure', 'ED Temp':'Vital_Sign Temperature Tympanic',
                                  'ED sats':'Vital_Sign Oxygen Saturation', 'ED dia_bp':'Vital_Sign Diastolic Blood Pressure',
                                  'charlson': 'Charlson Comorbidity Index'}, inplace=True)
    df_pt.rename(columns={'HB': 'Blood_Test HAEMOGLOBIN', 'WBC':'Blood_Test WHITE CELLS',
                                  'PLT':'Blood_Test PLATELETS', 'MCV':'Blood_Test MEAN CELL VOL.',
                                  'NEU':'Blood_Test NEUTROPHILS','HCT':'Blood_Test HAEMATOCRIT',
                                  'LYM':'Blood_Test LYMPHOCYTES', 'MON':'Blood_Test MONOCYTES',
                                  'EOS':'Blood_Test EOSINOPHILS', 'BAS':'Blood_Test BASOPHILS',
                                  'NA':'Blood_Test SODIUM', 'ALB':'Blood_Test ALBUMIN',
                                  'ALP':'Blood_Test ALK.PHOSPHATASE', 'ALT':'Blood_Test ALT',
                                  'U':'Blood_Test UREA', 'TBIL':'Blood_Test BILIRUBIN',
                                  'CR':'Blood_Test CREATININE', 'INR':'Blood_Test INR',
                                  'EGFR':'Blood_Test eGFR', 'K':'Blood_Test POTASSIUM',
                                  'PT':'Blood_Test Prothromb. Time', 'CRP':'Blood_Test CRP',
                                  'APTT':'Blood_Test APTT'}, inplace=True)
    df_pt.drop(['YY', 'MM', 'pulse', 'resp_rate','sats','sys_bp','dia_bp','TempR', 'RBC','MPV', 'ANRB', 'CA', 'COCA','P','GLHB', 'DDIM','MCHC', 'MCH'], axis=1, inplace=True)
    df_pt.rename(columns={'ICU_adm':'ICU'}, inplace=True)

    #Standardise PCR result (not supplied for pre-pandemic patients, inferred negatives)
    if 'pcr_result' in df_pt:
        df_pt.rename(columns={'pcr_result':'Covid-19 Positive'}, inplace=True)
    else:
        df_pt['Covid-19 Positive'] = 0

    #State admission state
    df_pt['Admission'] = 1

    #Standardise handling of extreme values
    df_pt = df_pt.replace({'<':''}, regex=True)
    df_pt = df_pt.replace({'>':''}, regex=True)

    #PUH issued a correction of COVID-19 status; these results are to supercede their prior extract.
    if PCR48hWindowFile:
        #Load correction file dependent on if Microsoft Excel or CSV
        if PCR48hWindowFile[-4:] == "xlsx":
            print ("Correction File Type: XLSX")
            pandemicEraUpdatedPCRResults = pd.read_excel(PCR48hWindowFile, dtype=object)
        elif PCR48hWindowFile[-3:] == "csv":
            pandemicEraUpdatedPCRResults = pd.read_csv(PCR48hWindowFile, dtype=object)
            print ("Correction File Type: CSV")

        #Replace Invalid PCR results with NaNs (i.e. an Invalid result should not be treated as a negative result)
        results_ids = {9: np.nan}
        pandemicEraUpdatedPCRResults['pcr_result48'].replace(results_ids, inplace=True)

        pandemicEraUpdatedPCRResults.rename(columns={'pcr_result48':'Covid-19 Positive'}, inplace=True)
        result = all(map(lambda x, y: x == y, df_pt['ClusterID'].astype('str'), pandemicEraUpdatedPCRResults['epis_sg2'].astype('str')))
        if result:
            print('Patient IDs in 48h PCR file are exactly equal')
            df_pt.head()
        else:
            print('Patient IDs in 48h PCR file are not equal')

        #Apply corrected PCR Result within 48h Window
        df_pt['Covid-19 Positive'] = pandemicEraUpdatedPCRResults['Covid-19 Positive']

    #Standardise datatype of Support Levels
    df_pt['Vital_Sign Delivery device used'].replace("L0",0,inplace = True)
    df_pt['Vital_Sign Delivery device used'].replace("L1",1,inplace = True)
    df_pt['Vital_Sign Delivery device used'].replace("L2",2,inplace = True)
    df_pt['Vital_Sign Delivery device used'].replace("L3",3,inplace = True)

    #Standardise Ethnicities
    df_pt['Ethnicity'] = ethnicityHarmoniser (df_pt['Ethnicity'])
    dataset = df_pt.reset_index(drop=True)

    #Save standardised dataset at local site.
    try:
        filename = siteName + " " + era + " Preprocessed Data.csv"
        dataset.to_csv("../processedData/"+filename)
    except:
        print ("Unable to save file")

    #Output n of file
    print(era+' n patients:\t\t{}'.format(len(dataset.count(axis='columns'))))

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
