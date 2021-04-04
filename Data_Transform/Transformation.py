
# ==============================================|*|| Required Libraries ||*|============================================
import pandas as pd
import os
from glob import glob
from datetime import datetime
# =======================================================| *** |========================================================

# ============================================|*|| User Defined Packages ||*|===========================================
from Data_Logs.Logs import Logs
# =======================================================| *** |========================================================

# ============================================|| Step 03 - Transforming ||==============================================
class Transform:
    def __init__(self, LogsPath, LogsFileName):
        self.Logs = Logs(LogsPath, LogsFileName)
        self.LogsList = list()

    def transformData(self, LocalPaths):
        self.LogsList.append(["Transform", f"Transforming files of GoodFiles", datetime.now(), "", "Started", ""])
        fileNameList = list()

        # ----------------------------------------------|| Load Good Files ||-------------------------------------------
        Local_CsvFiles = glob(os.path.join(LocalPaths["GoodFiles"], '*.csv'))
        self.LogsList.append(["Transform", f"Loading Good files", datetime.now(), "", f"{len(Local_CsvFiles)} are loaded successfully", ""])

        for file in Local_CsvFiles:
            file = file.replace("\\", '/')
            fileName = file.split("/")[-1]
            fileNameList.append(fileName)
            data = pd.read_csv(file)
            # ---------------------------->> Renaming columns values with Schema Col Values <<--------------------------
            data.columns = data.columns.str.replace('-', '')
            self.LogsList.append(["Transform", f"Rename of columnNames", datetime.now(), f"{fileName}",f"If any Column name has '-' replaced with empty -> ", ""])

            for column in data.columns:
                count = data[column][data[column] == '?'].count()
                if count != 0:
                    data[column] = data[column].replace('?', "'?'")
                    self.LogsList.append(["Transform", f"Adding quotes", datetime.now(), f"{fileName}", "", "Replaced ? with '?'", ""])
                data.to_csv(LocalPaths["TransformedFiles"] + fileName, index=False)
                self.LogsList.append(["Transform", f"Pushing File", datetime.now(), f"{fileName}", f"Transformed File is stored in 'TransformedFiles' folder", ""])

        self.LogsList.append(["Transform", f"Transforming files of GoodFiles", datetime.now(), "", "Completed", ""])
        self.Logs.storeLogs(self.LogsList)
# ======================================================| *** |=========================================================
