import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from InstanceSHAP.Data_preprocessing import PrepareData
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


class classifiers():
    def __init__(self):
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None

    def train_test(self):
        d = PrepareData()
        data = PrepareData.getdata(d)
        #data = data.iloc[:1000, :]
        x = data.drop('label', axis=1)
        y = data['label']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, stratify=y)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def lr_model(self):
        lr = LogisticRegression(random_state=123).fit(self.x_train, self.y_train)
        predicted_labels = lr.predict(self.x_test)
        return predicted_labels

    def rf_model(self):
        rf = RandomForestClassifier(random_state=123).fit(self.x_train, self.y_train)

        predicted_labels = rf.predict(self.x_test)
        return predicted_labels

    def model_evaluation(self):
        measures = ['Accuracy', 'Sensitivity', 'Specificity', 'f1_score', 'roc_auc_score', 'balanced_accuracy_score']
        ytrue = self.y_test
        predictions = {'lr_prediction': self.lr_model(), 'rf_prediction': self.rf_model()}
        eval_metrics = []
        models = list(predictions.keys())

        for p in range(len(models)):
            cm_lr = confusion_matrix(ytrue, list(predictions.values())[p])
            total_lr = sum(sum(cm_lr))
            #####from confusion matrix calculate accuracy
            eval_metrics.append(accuracy_score(ytrue, list(predictions.values())[p]))
            eval_metrics.append(cm_lr[0, 0] / (cm_lr[0, 0] + cm_lr[0, 1]))
            eval_metrics.append(cm_lr[1, 1] / (cm_lr[1, 0] + cm_lr[1, 1]))
            eval_metrics.append(f1_score(ytrue, list(predictions.values())[p]))
            eval_metrics.append(roc_auc_score(ytrue, list(predictions.values())[p]))
            eval_metrics.append(balanced_accuracy_score(ytrue, list(predictions.values())[p]))

        eval_df = pd.DataFrame(np.array(eval_metrics).reshape((len(models), 6)), index=models, columns=measures)
        return eval_df


if __name__ == '__main__':
    modeling = classifiers()
    modeling.train_test()
    df = modeling.model_evaluation()
    print(df)
