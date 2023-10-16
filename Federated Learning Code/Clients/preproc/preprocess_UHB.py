#Script to pre-process UHB data from source in to standardised format for FL

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

  #Load Dataset
  try:
      df_bm = pd.read_csv(pathToRawData)
  except FileNotFoundError:
      print("File "+pathToRawData+" not found.")
      time.sleep(10)
      exit()

  #Ensure dataset free of duplicates
  df_bm = df_bm.drop_duplicates()

  #Standardise the column names to common format
  OBI_df = pd.DataFrame([['StudyID', 'ClusterID', 'Encounter'], ['period', 'ArrivalDateTime', 'ClusterID'], ['Ethnicity', 'Ethnicity', 'Gender'], ['AGE ON PRESENTATION', 'Age', 'Ethnicity'], ['SEX (1 = FEMALE, 0 = MALE)', 'Gender', 'ArrivalDateTime'], ['AdmissionRespRate', 'Vital_Sign Respiratory Rate', 'Respiratory'], ['AdmissionHeartRate', 'Vital_Sign Heart Rate', 'CCI'], ['AdmissionBloodPressureSystolicBP', 'Vital_Sign Systolic Blood Pressure', 'CollectionDateTime_Covid'], ['AdmissionBloodPressureDiastolicBP', 'Vital_Sign Diastolic Blood Pressure', 'Covid-19 Positive'], ['AdmissionSpO2', 'Vital_Sign Oxygen Saturation', 'CollectionDateTime_Flu'], ['AdmissionTemperature', 'Vital_Sign Temperature Tympanic', 'Flu Positive'], ['AdmissionOxygenDeliveryDevice', 'Vital_Sign Delivery device used', 'CollectionDateTime_Other'], ['SARS-CoV-2 PCR', 'Covid-19 Positive', 'Other Positive'], ['PresentationHAEMOGLOBIN', 'Blood_Test HAEMOGLOBIN', 'Blood_Test ALBUMIN'], ['PresentationWHITE CELLS', 'Blood_Test WHITE CELLS', 'Blood_Test_Delta ALBUMIN'], ['PresentationPLATELETS', 'Blood_Test PLATELETS', 'Blood_Test ALK.PHOSPHATASE'], ['PresentationMEAN CELL VOL.', 'Blood_Test MEAN CELL VOL.', 'Blood_Test_Delta ALK.PHOSPHATASE'], ['PresentationNEUTROPHILS', 'Blood_Test NEUTROPHILS', 'Blood_Test_Delta ALT'], ['PresentationHAEMATOCRIT', 'Blood_Test HAEMATOCRIT', 'Blood_Test APTT'], ['PresentationLYMPHOCYTES', 'Blood_Test LYMPHOCYTES', 'Blood_Test BASOPHILS'], ['PresentationMONOCYTES', 'Blood_Test MONOCYTES', 'Blood_Test BILIRUBIN'], ['PresentationEOSINOPHILS', 'Blood_Test EOSINOPHILS', 'Blood_Test_Delta BILIRUBIN'], ['PresentationBASOPHILS', 'Blood_Test BASOPHILS', 'Blood_Test CREATININE'], ['PresentationSODIUM', 'Blood_Test SODIUM', 'Blood_Test EOSINOPHILS'], ['PresentationALBUMIN', 'Blood_Test ALBUMIN', 'Blood_Test_Delta EOSINOPHILS'], ['PresentationALK.PHOSPHATASE', 'Blood_Test ALK.PHOSPHATASE', 'Blood_Test HAEMATOCRIT'], ['PresentationALT', 'Blood_Test ALT', 'Blood_Test_Delta HAEMATOCRIT'], ['PresentationUREA', 'Blood_Test UREA', 'Blood_Test HAEMOGLOBIN'], ['PresentationBILIRUBIN', 'Blood_Test BILIRUBIN', 'Blood_Test_Delta HAEMOGLOBIN'], ['PresentationCREATININE', 'Blood_Test CREATININE', 'Blood_Test INR'], ['PresentationeGFR', 'Blood_Test eGFR', 'Blood_Test LYMPHOCYTES'], ['PresentationPOTASSIUM', 'Blood_Test POTASSIUM', 'Blood_Test_Delta LYMPHOCYTES'], ['PresentationCRP', 'Blood_Test CRP', 'Blood_Test_Delta MONOCYTES'], ['PresentationPOCT pC02', 'Blood_Gas pCO2 POC', 'Blood_Test NEUTROPHILS'], ['PresentationPOCT sO2', 'Blood_Gas O2 Sat (BG)', 'Blood_Test_Delta NEUTROPHILS'], ['PresentationPOCT pO2', 'Blood_Gas pO2 (BG)', 'Blood_Test PLATELETS'], ['PresentationPCT cBASE(Ecf)c', 'Blood_Gas BE Std (BG)', 'Blood_Test_Delta PLATELETS'], ['PresentationPCT CO3(P,st)c', 'Blood_Gas Bicarb (BG)', 'Blood_Test POTASSIUM'], ['PresentationPOCT Hctc', 'Blood_Gas Hct (BG)', 'Blood_Test_Delta POTASSIUM'], ['PresentationPOCT cGLU', 'Blood_Gas Glucose (BG)', 'Blood_Test Prothromb. Time'], ['PresentationPOCT cK+', 'Blood_Gas K+ (BG)', 'Blood_Test SODIUM'], ['PresentationPOCT cNA+', 'Blood_Gas Na+ (BG)', 'Blood_Test_Delta SODIUM'], ['PresentationPOCT cLAC', 'Blood_Gas cLAC (BG)', 'Blood_Test UREA'], ['PresentationPOCT cCA++', 'Blood_Gas Ca+ + (BG)', 'Blood_Test_Delta UREA'], ['PresentationProthromb. Time', 'Blood_Test Prothromb. Time', 'Blood_Test WHITE CELLS'], ['PresentationPOCT ctHb', 'Blood_Gas Hb (BG)', 'Blood_Test_Delta WHITE CELLS'], ['PresentationAPTT', 'Blood_Test APTT', 'Blood_Test_Delta eGFR'], ['Outcome (0=discharged, 1=ward admission, 2=ICU admission, 3=death)', 'ICU', 'Blood_Gas K+ (BG)']], columns=['BHD_cols_name', 'OXD_cols_intersect', 'OXD_cols_name'])
  OBI_df = OBI_df.dropna()
  BOI_sub = df_bm[OBI_df['BHD_cols_name']]
  b2o = {}
  for row in OBI_df.iterrows():
      b2o[row[1]['BHD_cols_name']]=row[1]['OXD_cols_intersect']
  del b2o['Outcome (0=discharged, 1=ward admission, 2=ICU admission, 3=death)']
  BOI_sub_converted = deepcopy(BOI_sub)

  #Standardise vital signs to the common format
  BOI_sub_converted['AdmissionOxygenDeliveryDevice'].replace(np.nan,0,inplace = True)
  BOI_sub_converted['AdmissionOxygenDeliveryDevice'].replace(['O2NASAL','O2VENTURI','O2TRACH','O2RESPIFLO', 'O2FACE'],1,inplace = True)
  BOI_sub_converted['AdmissionOxygenDeliveryDevice'].replace(['O2NOREMASK'],2,inplace = True)
  BOI_sub_converted['AdmissionOxygenDeliveryDevice'].replace(['O2NONINV','O2HFF','O2HFN'],3,inplace = True)
  #Replacement of free-text pre-pandemic and pandemic labels with example datetimes. NB: These are not absolute dates and should not be treated as such
  BOI_sub_converted.period.replace("pre-pandemic","2019-05-18 01:00:00",inplace = True)
  BOI_sub_converted.period.replace("pandemic","2020-5-8 01:00:00",inplace = True)

  #Standardise recording of gender
  BOI_sub_converted['SEX (1 = FEMALE, 0 = MALE)'].replace(1.0,"F",inplace = True)
  BOI_sub_converted['SEX (1 = FEMALE, 0 = MALE)'].replace(0.0,"M",inplace = True)

  # Standardise outcomes to the common format
  # NB: Label 4 in the file is intended to mean 3/death; this has been checked with data team at UHB.
  outcome_dummy = pd.get_dummies(BOI_sub['Outcome (0=discharged, 1=ward admission, 2=ICU admission, 3=death)'])
  outcome_dummy.columns = ["Discharged","Admission","ICU","death"]
  BOI_sub_converted = pd.concat([BOI_sub_converted,outcome_dummy],axis = 1)
  del BOI_sub_converted['Outcome (0=discharged, 1=ward admission, 2=ICU admission, 3=death)']

  #Standardise recording of COVID-19 test status to 1/0 (one-hot encoding)
  #Pre-pandemic patients will have NaN
  BOI_sub_converted['SARS-CoV-2 PCR'].replace(['Requested'],np.nan,inplace=True)
  BOI_sub_converted['SARS-CoV-2 PCR'].replace(['NOT Detected'],0.0,inplace=True)
  BOI_sub_converted['SARS-CoV-2 PCR'].replace(['**DETECTED**','***DETECTED***'],1.0,inplace=True)

  # Standardise admissions recording to common format
  #If a patient was ```Admitted to AMU```, ```Subsequently attended AMU```, or had an outcome of ```ICU``` or ```Admission```, they should all be considered to have been admitted.
  condition = [(df_bm['admitted AMU'] == 1)|(df_bm['subsequently attended AMU'] == 1)|(BOI_sub_converted['ICU'] == 1)]
  BOI_sub_converted['Admission'] = np.select(condition, [1])

  #Enact the above column name standardisation
  BOI_sub_converted.rename(columns = b2o, inplace = True)
  condition = [(BOI_sub_converted['Admission'] == 0),(BOI_sub_converted['Admission'] != 0)]
  BOI_sub_converted['EpisodeID'] = np.select(condition, [0,BOI_sub_converted['ClusterID']])
  BOI_sub_converted = BOI_sub_converted.replace({'<':''}, regex=True)
  BOI_sub_converted = BOI_sub_converted.replace({'>':''}, regex=True)

  #Standardse datatype for results
  BOI_sub_converted.loc[:,'Blood_Test HAEMOGLOBIN':'Blood_Test APTT'] = BOI_sub_converted.loc[:,'Blood_Test HAEMOGLOBIN':'Blood_Test APTT'].apply(pd.to_numeric, errors='raise')
  episodeid=BOI_sub_converted.pop('EpisodeID')
  BOI_sub_converted.insert(1, 'EpisodeID',episodeid)

  #Enforce exclusion criterion of removing patients with missing bloods
  bloods = BOI_sub_converted.loc[:,'Blood_Test HAEMOGLOBIN':'Blood_Test BASOPHILS'].columns
  bham_with_drops = BOI_sub_converted.dropna(subset = bloods, axis=0, thresh=5)
  BOI_sub_converted = BOI_sub_converted.dropna(subset = bloods, how='all')
  print('n patients:\t\t{}'.format(len(BOI_sub_converted.count(axis='columns'))))

  #Standardise ethnicities to the common format
  BOI_sub_converted['Ethnicity'] = ethnicityHarmoniser (BOI_sub_converted['Ethnicity'])

  #Save local data following the standardisation process
  BOI_sub_converted.to_csv("../processedData/UHB-CURIAL Processed Data.csv")
  dataset=BOI_sub_converted

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
