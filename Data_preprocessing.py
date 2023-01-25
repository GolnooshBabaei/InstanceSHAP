import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##settings
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)


class PrepareData:

    def __init__(self):
        self.num_features = None
        self.num_obs = None

    def import_data(self):
        """
        Import only the mostly common used columns in the literature
        Employment variables, length and title and also demographic variables are removed but later they can be more
        analyzed in the preprocessing.
        """
        data = pd.read_csv('C:/Users/assegnista/Downloads/Loan_status_2007-2020Q3.gzip',
                           usecols=['loan_amnt', 'term', 'grade', 'home_ownership',
                                    'annual_inc', 'purpose', 'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
                                    'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'loan_status'])

        self.num_obs = data.shape[0]
        self.num_features = data.shape[1]
        return data

    def feature_selection(self, data):
        """
        This function is used when whole columns of the original data are imported. In that case, using this
        function, variables for which nan ratio is less than 50% can be selected.
        """
        nans = pd.DataFrame(data.isna().sum(), columns=['count of nans'])
        nans['nan proportion'] = nans['count of nans'] / self.num_obs
        to_be_selected = list(nans[nans['nan proportion'] < 0.5].index)
        selected_data = data[to_be_selected]
        for i in to_be_selected:
            selected_data[i] = selected_data[i].fillna(selected_data[i].mean())

        return selected_data

    def feature_eng(self, data):
        data['fico'] = data[['fico_range_low', 'fico_range_high']].mean(axis=1)
        data.drop(['fico_range_low', 'fico_range_high'], axis=1, inplace=True)
        return data

    def encoding(self, data):
        """
        This function uses ratio encoding for categorical variables
        """

        # Binarize loan_status as the binary target variable
        data['label'] = np.where(data['loan_status'] == 'Charged Off', 1, 0)
        data.drop('loan_status', axis=1, inplace=True)
        # Encode other categorical columns
        cat_vars = data.select_dtypes(include='object').columns
        for i in cat_vars:
            failed_counts = pd.DataFrame(data.groupby(i)['label'].sum()).reset_index().rename(
                columns={'label': 'failed_counts'})
            tot = pd.DataFrame(data[i].value_counts()).reset_index().rename(columns={'index': i, i: 'total'})
            merged = pd.merge(failed_counts, tot, on=i)
            merged['{}_pd'.format(i)] = merged['failed_counts'] / merged['total']
            data = pd.concat([data, pd.merge(data, merged, on=i, how='left')['{}_pd'.format(i)]], axis=1)
        data.drop(cat_vars, axis=1, inplace=True)
        return data

    def getdata(self):
        df_ = self.import_data()
        df_1 = self.encoding(df_)
        df_2 = self.feature_eng(df_1)
        df_3 = self.feature_selection(df_2)

        return df_3


if __name__ == '__main__':
    cl = PrepareData()
    df = cl.getdata()

    print(df.head())
    print(df.shape)
    print(df.isna().sum())
