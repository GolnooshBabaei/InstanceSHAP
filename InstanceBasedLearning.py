import shap
from sklearn.model_selection import train_test_split
from InstanceSHAP.Data_preprocessing import PrepareData
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


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
        data = PrepareData.getdata(data_class).iloc[:1000, :]
        ## To reduce the computation time, I use 0.3 of each class available in the dataset
        data = data.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.3))
        x = data.drop('label', axis=1)
        y = data['label']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, stratify=y)
        self.x_train.reset_index(inplace=True)
        self.x_test.reset_index(inplace=True)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def get_model_predictions(self):
        rf = RandomForestClassifier(random_state=123).fit(self.x_train, self.y_train)

        self.pd_test = rf.predict_proba(self.x_test)[:, 1]
        self.pd_train = rf.predict_proba(self.x_train)[:, 1]
        sigma_predictedvalue_train = np.std(self.pd_train)
        return self.pd_test, self.pd_train, sigma_predictedvalue_train

    def calc_kernel(self, u):
        kernel = (1 / np.sqrt(2 * np.pi)) * (1 / np.exp(0.5 * (u ** 2)))
        return kernel

    def find_similarities(self):
        d = []
        for i in range(len(self.x_train)):
            d.append(self.pd_train[i] - self.pd_test)
        d = pd.DataFrame(d)
        return d

    def run_model(self):
        hzero = ((4 / (3 + len(self.x_train))) ** 0.2) * self.get_model_predictions()[2]
        hratio = np.arange(0.25, 1.5, 0.1)

        gini_ordinal = []
        gini_instancebased = []
        optimal_h_set = []
        Number_of_similar_observations_in_the_InstanceSHAP = []

        for r in range(2):

            ret = xtrain['interest']
            final_cv_ = []

            ##create d matrix
            d = pd.DataFrame(euclidean_distances(self.x_train, self.x_test))

            ## find cv for all possible values of h
            for h in hratio:
                h_ = hzero * h
                final_cv = []
                cv_mat = pd.DataFrame(np.zeros((len(self.x_train), len(self.x_train))))
                r_cv_mat = cv_mat.copy()

                for j in range(len(self.x_train)):
                    for k in range(len(self.x_train)):
                        cv_mat.iloc[j, k] = self.calc_kernel((self.pd_train[j] - self.pd_train[k]) / h_)
                        r_cv_mat.iloc[j, k] = cv_mat.iloc[j, k] * ret[k]

                    final_cv.append(((r_cv_mat.iloc[j, :].sum() / cv_mat.iloc[j, :].sum()) - ret[j]) ** 2)

                final_cv_.append(np.sum(final_cv) / len(cv_mat))

            idx_mincv = np.argmin(final_cv_)
            optimalh = hratio[idx_mincv] * hzero
            optimal_h_set.append(optimalh)
            kernel_d = d.apply(lambda x: self.calc_kernel(x / h))
            sum_rows = kernel_d.sum(axis=1)
            ##find weights between train and test data
            weights = pd.DataFrame(np.zeros((len(self.x_train), len(self.x_test))))
            for i in range(len(self.x_train)):
                for j in range(len(self.x_test)):
                    weights.iloc[i, j] = kernel_d.iloc[i, j] / sum_rows[i]

            weight_treshold = np.mean(weights)

            similarbackgrounddata = self.x_train.loc[
                (weights.loc[weights.where(weights > weight_treshold).any(axis=1)]).index]

            ##find shapley values for two cases, one with the whole train data as the background data and one with the
            # background dataset including only similar observations from train data

            #ordinal_shapleyvalues = self.find_explanations(mymodel, xtrain_ml, xtest_ml)
            #instancebased_shapleyvalues = self.find_explanations(mymodel, similarbackgrounddata, xtest_ml)
            #Number_of_similar_observations_in_the_InstanceSHAP.append(instancebased_shapleyvalues.shape[0])

            ##calculate concentration measure such as Gini index to compare both approaches
            #gini_ordinal.append(indices.gini(ordinal_shapleyvalues))
            #gini_instancebased.append(indices.gini(instancebased_shapleyvalues))
        return similarbackgrounddata

    def find_explanations(self, trainedmodel, backgrounddata, xtest):
        if len(backgrounddata) > 0:
            explainer = shap.TreeExplainer(model=trainedmodel, data=backgrounddata)
            shapleyvalues = explainer.shap_values(xtest)
            shapleyvalues_df = pd.DataFrame(shapleyvalues[1], columns=xtest.columns)
            global_shapleyvalues = np.mean(np.abs(shapleyvalues_df), axis=0)
        else:
            explainer = shap.TreeExplainer(model=model, data=pd.DataFrame(np.zeros((1, xtrain_ml.shape[1]))))
            shapleyvalues = explainer.shap_values(xtest)
            shapleyvalues_df = pd.DataFrame(shapleyvalues[1], columns=xtest.columns)
            global_shapleyvalues = np.mean(np.abs(shapleyvalues_df), axis=0)
        return global_shapleyvalues


if __name__ == '__main__':
    c = INSTANCEBASEDSHAP()
    data = c.read_data()
    model = c.get_model_predictions()
    d = c.find_similarities()

    print(d.head())
    print(d.shape)
