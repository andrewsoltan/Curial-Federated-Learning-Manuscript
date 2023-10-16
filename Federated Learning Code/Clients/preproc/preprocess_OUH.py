#Script to pre-process OUH data from source in to standardised format for FL

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
        dataset = pd.read_csv(pathToRawData)
    except FileNotFoundError:
        print("File "+pathToRawData+" not found.")
        time.sleep(10)
        exit()

    #Pre-processed pathway already including selections
    print('n patients:\t\t{}'.format(len(dataset.count(axis='columns'))))

    #Standardise Ethnicities to common format
    dataset['Ethnicity'] = ethnicityHarmoniser (dataset['Ethnicity'])

    #Save standardise form of dataset
    dataset.to_csv("../processedData/OUH-CURIAL Processed Data.csv")
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
