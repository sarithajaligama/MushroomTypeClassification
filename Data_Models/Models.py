# ==============================================|*|| Required Libraries ||*|============================================
from sklearn.cluster import KMeans
from kneed import KneeLocator

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score
import joblib
import os
from datetime import datetime
# =======================================================| *** |========================================================

# ============================================|*|| User Defined Packages ||*|===========================================
from Data_Logs.Logs import Logs
from Global_Variables.Global_Variables import GlobalVariablesPaths
# =======================================================| *** |========================================================

# =============================================|| Step 06 - Preprocess ||===============================================
class Model:
    def __init__(self, LogsPath, LogsFileName):
        self.Logs = Logs(LogsPath, LogsFileName)
        self.LogsList = list()
        GVP = GlobalVariablesPaths()
        self.LocalTrainingPaths = GVP.LocalTrainingPaths

    def trainingModel(self, LocalPaths, FileNames):
        try:
            self.LogsList.append(["Training Model", f"Model training", datetime.now(), "", "Started", ""])
            # ------------------------------------------------->> Load Data <<------------------------------------------
            # 'Temp_Files\Training\SingleFile\SingleFile_Scale.csv'
            FilePath = LocalPaths['SingleFile'] + FileNames["Input_FileName"]
            data = pd.read_csv(FilePath)
            self.LogsList.append(["Training Model", f"Loading scaled data", datetime.now(), f"{FileNames['Input_FileName']}", "Loaded", ""])

            # ---------------------------->> Split 'Predictors and 'Response' <<----------------------------
            X = data.drop(['class'], axis=1)
            Y = data['class']
            # ---------------------------->> Find and Add clusters to the 'Predictors' <<----------------------------
            # --------------->> Elbow PLot
            self.LogsList.append(["Training Model", f"k-means Cluster", datetime.now(), "", "Started", ""])
            wcss = [KMeans(n_clusters=i, init='k-means++', random_state=42).fit(X).inertia_ for i in range(1, 11)]
            plt.plot(range(1, 11), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.savefig(LocalPaths["Clusters"] + "K-Means_Elbow.PNG")
            self.LogsList.append(["Training Model", f"k-means Cluster", datetime.now(), "", f"Save cluster fig path: {LocalPaths['Clusters']+'K-Means_Elbow.PNG'}", ""])

            # --------------->> Knee Locator
            number_of_Clusters = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing').knee
            self.LogsList.append(["Training Model", f"k-means Cluster", datetime.now(), "", f"Number of clusters: {number_of_Clusters}", "" ])
            kmeans = KMeans(n_clusters=number_of_Clusters, init='k-means++', random_state=42)
            X["Clusters"] = kmeans.fit_predict(X)
            X['Labels'] = Y
            self.LogsList.append(["Training Model", f"k-means Cluster", datetime.now(), "", "Applying k-means cluster on Predictors data", ""])

            # --------------->> Storing KMeans cluster in Local
            joblib.dump(kmeans, LocalPaths["Clusters"] + "KMeans_Cluster.sav")
            self.LogsList.append(["Training Model", f"k-means Cluster", datetime.now(), "", "Completed", f"Storing k-means cluster path: {LocalPaths['Clusters'] + FileNames['Cluster_FileName']}"])

            # ----------------------------------------------->> Finding Best Model <<---------------------------------------
            self.LogsList.append(["Training Model", "Best Model Finding", datetime.now(), "", "Started", ""])
            for i in X['Clusters'].unique():
                # --------------->> Divide Clusters
                cluster_data = X[X['Clusters'] == i]
                self.LogsList.append(["Training Model", "Best Model Finding", datetime.now(), "", f"Extracting data of cluster-{i}", ""])
                X_cluster = cluster_data.drop(['Labels', 'Clusters'], axis=1)
                Y_cluster = cluster_data['Labels']
                X_train, X_test, Y_train, Y_test = train_test_split(X_cluster, Y_cluster, test_size=1 / 3, random_state=355)
                self.LogsList.append(["Training Model", "Best Model Finding", datetime.now(), "", f"Train_Test_Split of cluster-{i}", ""])

                # ----------------------------------------------->> KNN <<----------------------------------------------
                self.LogsList.append(["Training Model", "KNN model training", datetime.now(), "", "Started", ""])
                # --------------->> Getting best parameters of KNN
                param_grid_knn = {'algorithm': ['ball_tree', 'kd_tree', 'brute'], 'leaf_size': [10, 17, 24, 28, 30, 35],
                                  'n_neighbors': [4, 5, 8, 10, 11], 'p': [1, 2]}
                grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn, cv=5, verbose=3)
                grid.fit(X_train, Y_train)
                self.LogsList.append(["Training Model", "KNN model training - Best params", datetime.now(), "", f"algorithm:{grid.best_params_['algorithm']}, leaf_size:{grid.best_params_['leaf_size']}, n_neighbors:{grid.best_params_['n_neighbors']}, p:{grid.best_params_['p']}", ""])

                # --------------->> creating a new model with the best parameters
                KNN = KNeighborsClassifier(algorithm    = grid.best_params_['algorithm'],
                                           leaf_size    = grid.best_params_['leaf_size'],
                                           n_neighbors  = grid.best_params_['n_neighbors'],
                                           p            = grid.best_params_['p'])
                # --------------->> training the mew model
                KNN.fit(X_train, Y_train)
                # --------------->> Predictions using the KNNModel
                Y_test_predict_KNN = KNN.predict(X_test)
                self.LogsList.append(["Training Model", "KNN model training", datetime.now(), "", f"Model is trained for cluster-{i}", ""])

                # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                if len(Y_test.unique()) == 1:
                    KNN_score = accuracy_score(Y_test, Y_test_predict_KNN)
                else:
                    KNN_score = roc_auc_score(Y_test, Y_test_predict_KNN)  # AUC for KNN
                self.LogsList.append(["Training Model", "KNN model training", datetime.now(), "", f"KNN Score for cluster-{i}: {KNN_score}", ""])
                self.LogsList.append(["Training Model", "KNN model training", datetime.now(), "", "Comleted", ""])
                # -------------------------------------------------->> RFC <<-----------------------------------------------
                self.LogsList.append(["Training Model", "RFC model training", datetime.now(), "", "Started", ""])
                param_grid_rfc = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                                  "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}
                grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid_rfc, cv=5, verbose=3)
                grid.fit(X_train, Y_train)
                self.LogsList.append(["Training Model", "RFC model training - Best params", datetime.now(), "", f"criterion:{grid.best_params_['criterion']}, max_depth:{grid.best_params_['max_depth']}, max_features:{grid.best_params_['max_features']}, n_estimators:{grid.best_params_['n_estimators']}", ""])

                RFC = RandomForestClassifier(n_estimators = grid.best_params_['n_estimators'],
                                             criterion    = grid.best_params_['criterion'],
                                             max_depth    = grid.best_params_['max_depth'],
                                             max_features = grid.best_params_['max_features'])
                RFC.fit(X_train, Y_train)
                # --------------->> prediction using the Random Forest Algorithm
                Y_test_predict_RFC = RFC.predict(X_test)
                self.LogsList.append(["Training Model", "RFC model training", datetime.now(), "", f"Model is trained for cluster-{i}", ""])
                self.LogsList.append(["Training Model", "RFC model training", datetime.now(), "", "Comleted", ""])

                if len(Y_test.unique()) == 1:
                    # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                    RFC_score = accuracy_score(Y_test, Y_test_predict_RFC)
                else:
                    RFC_score = roc_auc_score(Y_test, Y_test_predict_RFC)  # AUC for Random Forest
                self.LogsList.append(["Training Model", "RFC model training", datetime.now(), "", f"RFC Score for cluster-{i}: {RFC_score}", ""])

                # --------------->> comparing the two models
                if (RFC_score < KNN_score):
                    file = LocalPaths["BestModels"] + FileNames["BestModel_FileName"] + f"KNN{str(i)}.sav"
                    model = KNN
                else:
                    file = LocalPaths["BestModels"] +FileNames["BestModel_FileName"] + f"RFC{str(i)}.sav"
                    model = RFC
                self.LogsList.append(["Training Model", "Best Model Finding", datetime.now(), "", f"Best Model for cluster-{i} -> {str(model)}", ""])

                # --------------->> Storing best model for each cluster data
                joblib.dump(model, file)
                self.LogsList.append(["Training Model", "Best Model Finding", datetime.now(), "", f"Storing {str(model)} model at: {file}", ""])
            self.LogsList.append(["Training Model", f"Best Model Finding", datetime.now(), "", "Completed", ""])
            self.LogsList.append(["Training Model", f"Model training", datetime.now(), "", "Completed", ""])
            self.Logs.storeLogs(self.LogsList)
        except Exception as e:
            self.LogsList.append(["Training Model", f"Model training", datetime.now(), "", f"Error: {e}", ""])
            self.Logs.storeLogs(self.LogsList)



    def predictingModel(self, LocalPaths, FileNames):
        try:
            self.LogsList.append(["Prediction", f"Predicting Outcome", datetime.now(), "", "Started", ""])
            data = pd.read_csv(LocalPaths['SingleFile'] + FileNames["Input_FileName"])
            self.LogsList.append(["Prediction", f"Loading Scaled data", datetime.now(), "", f"{FileNames['Input_FileName']} is loaded", ""])
            data_predict = pd.DataFrame(columns=list(data.columns) + ["Labels"], dtype='object')

            # --------------->> Clustering
            self.LogsList.append(["Prediction", f"k-means cluster", datetime.now(), "","Started", ""])
            kmeans_cluster = joblib.load(self.LocalTrainingPaths['Clusters'] + FileNames['Cluster_FileName'])
            self.LogsList.append(["Prediction", f"Loading k-means cluster", datetime.now(), "", f"{FileNames['Cluster_FileName']} is loaded", ""])
            data['Clusters'] = kmeans_cluster.fit_predict(data)
            self.LogsList.append(["Prediction", f"Finding clusters", datetime.now(), "", f"Spliting data into clusters based on training clusters", ""])
            self.LogsList.append(["Prediction", f"k-means cluster", datetime.now(), "","Completed", ""])

            listOfBestModels = os.listdir(self.LocalTrainingPaths["BestModels"])
            for i in data['Clusters'].unique():
                cluster_data = data[data['Clusters'] == i]
                self.LogsList.append(["Prediction", f"Extracting cluster", datetime.now(), "", f"Extracted data of cluster-{i}", ""])
                # --------------->> Loading suitable model for the above Cluster
                for modelName in listOfBestModels:
                    if i == int(modelName[-5]):
                        model = joblib.load(self.LocalTrainingPaths["BestModels"] + modelName)
                        self.LogsList.append(["Prediction", f"Loading Model", datetime.now(), "", f"Suitable model for cluster-{i} is loaded", ""])

                        # --------------->> Predicting the Response
                cluster_data.drop(['Clusters'], axis=1, inplace=True)  # Drop the "Cluster" column
                cluster_data["Labels"] = model.predict(cluster_data)
                self.LogsList.append(["Prediction", f"Loading Model", datetime.now(), "", f"Predicted response for data of cluster-{i}", ""])
                data_predict = pd.concat([data_predict, cluster_data])
                self.LogsList.append(["Prediction", f"Combining cluster data", datetime.now(), "", f"cluster-{i} is merged with cluster-{i-1}", ""])

            data_predict.to_csv(LocalPaths["PredictedFiles"] + FileNames["Prediction_FileName"], index=False)
            self.LogsList.append(["Prediction", f"Storing data", datetime.now(), "", f"Save prediction file: {FileNames['Prediction_FileName']}", ""])
            # ---------------------------------------------------| *** |----------------------------------------------------
            self.LogsList.append(["Prediction", f"Predicting Outcome", datetime.now(), "", "Completed", ""])
            self.Logs.storeLogs(self.LogsList)

        except Exception as e:
            self.LogsList.append(["Prediction", f"Predicting Outcome", datetime.now(), "", f"Error: {e}", ""])
            self.Logs.storeLogs(self.LogsList)
# =========================================================| *** |======================================================