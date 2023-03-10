import shap
from sklearn.model_selection import train_test_split
from InstanceSHAP.Data_preprocessing import PrepareData
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from concentrationMetrics import Index
import random


class INSTANCEBASEDSHAP:
    def __init__(self):
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None
        self.pd_test = None
        self.pd_train = None

    def read_data(self):
        data_class = PrepareData()
        data = PrepareData.getdata(data_class)
        data = data.iloc[:1000, :]
        ## To reduce the computation time, I use 0.3 of each class available in the dataset
        #data = data.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.5))
        x = data.drop('label', axis=1)
        y = data['label']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, stratify=y)
        self.x_train.reset_index(inplace=True, drop=True)
        self.x_test.reset_index(inplace=True, drop=True)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def get_model_predictions(self):
        rf = RandomForestClassifier(random_state=123).fit(self.x_train, self.y_train)

        self.pd_test = rf.predict_proba(self.x_test)[:, 1]
        self.pd_train = rf.predict_proba(self.x_train)[:, 1]
        sigma_predictedvalue_train = np.std(self.pd_train)
        return self.pd_test, self.pd_train, rf

    def find_similar_obs(self):
        dist = euclidean_distances(self.x_train, self.x_test)
        dist = pd.DataFrame(dist)
        scaler = MinMaxScaler().fit(dist)
        d_scaled = pd.DataFrame(scaler.transform(dist), columns=dist.columns)
        weights = pd.DataFrame(np.ones((len(self.x_train), len(self.x_test)))) - d_scaled
        weight_treshold = np.mean(weights.values.flatten())
        indices = []
        for i in range(len(self.x_test)):
            indices.append([index for index, item in enumerate(weights[i]) if item == 1])
        flat_indices = [x for l in indices for x in l]
        similarbackgrounddata = self.x_train.iloc[flat_indices, :]
        return similarbackgrounddata

    def find_explanations(self, backgrounddata= None):
        rf_model = self.get_model_predictions()[2]
        explainer = shap.TreeExplainer(model=rf_model, data=backgrounddata)
        shapleyvalues = explainer.shap_values(self.x_test)
        shapleyvalues_df = pd.DataFrame(shapleyvalues[1], columns=self.x_test.columns)
        global_shapleyvalues = np.mean(np.abs(shapleyvalues_df), axis=0)
        return global_shapleyvalues

    def Compare_explanations(self):
        indices = Index()
        rf_model = self.get_model_predictions()[2]
        exp_classic = shap.TreeExplainer(model=rf_model)
        Classic_shapleyvalues = exp_classic.shap_values(self.x_test)
        classic_shapleyvalues_df = pd.DataFrame(Classic_shapleyvalues[1], columns=self.x_test.columns)
        global_classicshapleyvalues = np.mean(np.abs(classic_shapleyvalues_df), axis=0)
        exp_instance = shap.TreeExplainer(model=rf_model, data=self.find_similar_obs())
        Instancebased_shapleyvalues = exp_instance.shap_values(self.x_test)
        instance_shapleyvalues_df = pd.DataFrame(Instancebased_shapleyvalues[1], columns=self.x_test.columns)
        global_instanceshapleyvalues = np.mean(np.abs(instance_shapleyvalues_df), axis=0)
        ##calculate concentration measure such as Gini index to compare both approaches
        gini_classic = indices.gini(global_classicshapleyvalues)
        gini_instancebased = indices.gini(global_instanceshapleyvalues)
        return gini_classic, gini_instancebased


if __name__ == '__main__':
    c = INSTANCEBASEDSHAP()
    data = c.read_data()
    #print(data[0].head())
    #print(data[1].head())
    #print(data[0].shape)
    #print(data[1].shape)
    #model = c.get_model_predictions()
    #similar_rows = c.find_similar_obs()
    #shap_vals = c.find_explanations(model[3], similar_rows, data[1])
    gini_vals = c.Compare_explanations()
    print('Gini index for classic SHAP', gini_vals[0])
    print('Gini index for InstanceSHAP', gini_vals[1])
    print('InstanceSHAP gini index is {} higher than that of for Classic SHAP'.format(gini_vals[1]-gini_vals[0]))
    #print('number of rows in new extracted data: ', similar_rows.shape)
    #print('number of rows in train data: ', data[0].shape)
