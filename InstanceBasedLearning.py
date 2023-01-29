import shap
from sklearn.model_selection import train_test_split
from InstanceSHAP.Data_preprocessing import PrepareData
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

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
        data = data.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.5))
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
        return self.pd_test, self.pd_train, sigma_predictedvalue_train, rf

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

    def find_explanations(self, trainedmodel, backgrounddata, xtest):
        if len(backgrounddata) > 0:
            explainer = shap.TreeExplainer(model=trainedmodel, data=backgrounddata)
            shapleyvalues = explainer.shap_values(xtest)
            shapleyvalues_df = pd.DataFrame(shapleyvalues[1], columns=xtest.columns)
            global_shapleyvalues = np.mean(np.abs(shapleyvalues_df), axis=0)
        else:
            explainer = shap.TreeExplainer(model=model, data=pd.DataFrame(np.zeros((1, self.x_train.shape[1]))))
            shapleyvalues = explainer.shap_values(xtest)
            shapleyvalues_df = pd.DataFrame(shapleyvalues[1], columns=xtest.columns)
            global_shapleyvalues = np.mean(np.abs(shapleyvalues_df), axis=0)
        return global_shapleyvalues


if __name__ == '__main__':
    c = INSTANCEBASEDSHAP()
    data = c.read_data()
    model = c.get_model_predictions()
    similar_rows = c.find_similar_obs()
    shap_vals = c.find_explanations(model[3], similar_rows, data[1])
    print(shap_vals)
    #print('number of rows in new extracted data: ', similar_rows.shape)
    #print('number of rows in train data: ', data[0].shape)
