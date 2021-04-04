# ==============================================|*|| Required Libraries ||*|============================================
from glob import glob
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn_pandas import CategoricalImputer
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# =======================================================| *** |========================================================

# ============================================|*|| User Defined Packages ||*|===========================================
from Data_Logs.Logs import Logs
from Global_Variables.Global_Variables import GlobalVariablesPaths
# =======================================================| *** |========================================================

# =============================================|| Step 06 - Preprocess ||===============================================

class Preprocess:
    def __init__(self, LogsPath, LogsFileName):
        self.Logs = Logs(LogsPath, LogsFileName)
        self.LogsList = list()
        GVP = GlobalVariablesPaths()
        self.LocalTrainingPaths = GVP.LocalTrainingPaths

    def preprocesTraningsData(self, LocalPaths, FileNames):
        self.LogsList.append(["Preprocess", "Preprocessing Training data", datetime.now(), "", "Started", ""])
        data = pd.read_csv(LocalPaths['SingleFile'] + FileNames['Single_FileName'])
        self.LogsList.append(["Preprocess", "Loading File", datetime.now(), f"{FileNames['Single_FileName']}", "Data frame created with the file", ""])

        # -------------------------------->> Storing Missing values into Csv file<<-------------------------------------
        Missing_Df = pd.DataFrame(columns=data.columns, dtype='object')
        for col in data:
            data[col] = data[col].replace("'?'", np.nan)
        Missing_Df = Missing_Df.append(data.isnull().sum(), ignore_index=True)
        self.LogsList.append(["Preprocess", f"Missing values", datetime.now(), f"{data}",f"Missing values of columns for {data} are stored", ""])
        Missing_Df.to_csv(LocalPaths["LogFiles"] + FileNames["MissingValues_FileName"])
        self.LogsList.append(["Preprocess", f"Pushing File", datetime.now(), "", f"Missing values of columns for each file is stored in 'MissingValues_FileName' folder", ""])

        # ------------------------------------------>> Impute Missing values <<-----------------------------------------
        for col in data:
            if data[col].isnull().sum() > 0:
                data[col] = CategoricalImputer().fit_transform(data[col])
            self.LogsList.append(["Preprocess", f"Imputing missing values", datetime.now(), f"{data}", f"Missing values in {data}->{col} : {data[col].isnull().sum()}",f"Imputed with {CategoricalImputer()}"])

            # ---------------------------->> FeatureSelection : Removing unnecessary columns <<-------------------------
            if len(data[col].unique()) == 1:
                self.LogsList.append(["Preprocess", "Removing unnecessary columns", datetime.now(), f"{FileNames['Single_FileName']}",f"{list(data[col].unique()) == 1}columns are removed", ""])
                data.drop(col, axis=1, inplace=True)

        # ----------------------------------->> Convert categorical into numerical <<-----------------------------------
        label_encoder = preprocessing.LabelEncoder()
        data['class'] = pd.Series(label_encoder.fit_transform(y=data['class']))
        for col in data:
            if data[col].dtype == 'object':
                data = pd.get_dummies(data, columns=[col], drop_first=True)
        self.LogsList.append(["Preprocess", f"Convert categorical into numerical", datetime.now(), f"{data}", f"Categorical column : {data}->{col}", "Convert to Numerical"])

        # ----------------------------->> Saving Scaled values into a ScaledFile <<-------------------------------------
        Clean_Data = pd.DataFrame(columns=data.columns, dtype='int64')
        Clean_Data = pd.concat([Clean_Data, data])
        Clean_Data.to_csv(LocalPaths['SingleFile'] + FileNames['Input_FileName'], index=False)
        self.LogsList.append(["Preprocess", "Storing File", datetime.now(), f"{FileNames['Single_FileName']}", f"InputFile is stored in SingleFiles fodler", ""])

        self.LogsList.append(["Preprocess", "Preprocessing Training data", datetime.now(), "", "Completed", ""])
        self.Logs.storeLogs(self.LogsList)
        # ======================================================| *** |=================================================

    def preprocesPredictingsData(self, LocalPaths, FileNames):
        self.LogsList.append(["Preprocess", "Preprocessing Training data", datetime.now(), "", "Started", ""])
        data = pd.read_csv(LocalPaths['SingleFile'] + FileNames['Single_FileName'])
        self.LogsList.append(["Preprocess", "Loading File", datetime.now(), f"{FileNames['Single_FileName']}", "Data frame created with the file", ""])

        # -------------------------------->> Storing Missing values into Csv file<<-------------------------------------
        Missing_Df = pd.DataFrame(columns=data.columns, dtype='object')
        for col in data:
            data[col] = data[col].replace("'?'", np.nan)
        Missing_Df = Missing_Df.append(data.isnull().sum(), ignore_index=True)
        self.LogsList.append(["Preprocess", f"Missing values", datetime.now(), f"{data}",f"Missing values of columns for {data} are stored", ""])
        Missing_Df.to_csv(LocalPaths["LogFiles"] + FileNames["MissingValues_FileName"])
        self.LogsList.append(["Preprocess", f"Pushing File", datetime.now(), "", f"Missing values of columns for each file is stored in 'MissingValues_FileName' folder", ""])

        # ------------------------------------------>> Impute Missing values <<-----------------------------------------
        for col in data:
            if data[col].isnull().sum() > 0:
                data[col] = CategoricalImputer().fit_transform(data[col])
            self.LogsList.append(["Preprocess", f"Imputing missing values", datetime.now(), f"{data}", f"Missing values in {data}->{col} : {data[col].isnull().sum()}",f"Imputed with {CategoricalImputer()}"])

        # ------------------------------>> FeatureSelection : Removing unnecessary columns <<---------------------------
            if len(data[col].unique()) == 1:
                self.LogsList.append(["Preprocess", "Removing unnecessary columns", datetime.now(), f"{FileNames['Single_FileName']}",f"{list(data[col].unique()) == 1}columns are removed", ""])
                data.drop(col, axis=1, inplace=True)

        # ----------------------------------->> Convert categorical into numerical <<-----------------------------------
        for col in data:
            if data[col].dtype == 'object':
                data = pd.get_dummies(data, columns=[col], drop_first=True)
        self.LogsList.append(["Preprocess", f"Convert categorical into numerical", datetime.now(), f"{data}", f"Categorical column : {data}->{col}", "Convert to Numerical"])

        # ----------------------------->> Saving Scaled values into a ScaledFile <<-------------------------------------
        Clean_Data = pd.DataFrame(columns=data.columns, dtype='int64')
        Clean_Data = pd.concat([Clean_Data, data])
        Clean_Data.to_csv(LocalPaths['SingleFile'] + FileNames['Input_FileName'], index=False)
        self.LogsList.append(["Preprocess", "Storing File", datetime.now(), f"{FileNames['Single_FileName']}", f"InputFile is stored in SingleFiles fodler", ""])

        self.LogsList.append(["Preprocess", "Preprocessing Training data", datetime.now(), "", "Completed", ""])
        self.Logs.storeLogs(self.LogsList)
# =========================================================| *** |======================================================