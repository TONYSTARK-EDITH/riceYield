import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, Normalizer, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import asyncio


class saiyan_model:
    def __init__(self, file_name, print_option_precision=2, is_print=False, r_state=0, train_size=0.8, test_size=0.2,
                 re_scale=False, std_scale=False, nor_scale=False, encode_col_data=False, which_col=-1,
                 remainder='passthrough', degree=4, kernel='rbf', n_estimators=10, r_state_for_rf=0,
                 encode_label=False, random_state_for_dtr=0, is_classification=False, feature_scale_x=True,
                 feature_scale_y=True, is_confusion_matrix=False, estimators_n=100):
        self.is_confusion_matrix = is_confusion_matrix
        self.rfc_model = RandomForestClassifier(n_estimators=estimators_n, criterion='entropy')
        self.dtc_model = DecisionTreeClassifier(criterion="entropy", random_state=0)
        self.nb_model = GaussianNB()
        self.svc_model = SVC(kernel='rbf', random_state=0)
        self.knn_classifier_model = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
        self.log_reg_model = LogisticRegression(random_state=0)
        self.random_state_for_dtr = random_state_for_dtr
        self.r_state_for_rf = r_state_for_rf
        self.n_estimators = n_estimators
        self.rf_model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.r_state_for_rf)
        self.dtr_model = DecisionTreeRegressor(random_state=self.random_state_for_dtr)
        self.is_now_changed = False
        self.kernel = kernel
        self.X = None
        self.y = None
        self.lr_model = LinearRegression()
        self.file = file_name
        self.dataset = pd.read_csv(self.file)
        self.print_option_precision = print_option_precision
        self.is_print = is_print
        self.r_state = r_state
        self.train_size = train_size
        self.test_size = test_size
        self.re_scale = re_scale
        self.std_scale = std_scale
        self.nor_scale = nor_scale
        self.encode_col_data = encode_col_data
        self.which_col = which_col
        self.encode_label = encode_label
        self.remainder = remainder
        self.degree = degree
        self.label_encoder = LabelEncoder()
        self.svr_model = SVR(kernel=self.kernel)
        self.pr_model = PolynomialFeatures(degree=self.degree)
        self.is_classifier = is_classification
        self.feature_x = feature_scale_x
        self.feature_y = feature_scale_y
        self.regression_saiyan_model = {}
        self.classification_saiyan_model = {}
        if self.encode_col_data:
            if self.which_col == -1:
                raise AttributeError("Column should be provided in order to encode the column")
        if self.re_scale:
            if not self.std_scale and not self.nor_scale:
                self.std_scale = True
        self.__split__()

    def __split__(self):
        self.X, self.y = self.dataset.iloc[:, 1:-1].values, self.dataset.iloc[:, -1].values
        self.X = self.X.astype(int)
        self.y = self.y.astype(int)
        print(self.X)
        self.y = self.y.reshape(-1, 1)
        if self.encode_col_data:
            self.encode_categorical_data()
        if self.encode_label:
            self.encode_label_dependent_variable()
        if self.train_size != 0.8:
            self.test_size = 1 - self.train_size
        if self.train_size == 1 or self.test_size == 1:
            raise AttributeError("train_size or test_size should not be one")
        if self.train_size < 0 or self.test_size < 0:
            raise AttributeError("train_size or test_size should not be less than zero")
        if self.test_size != 0.2:
            self.train_size = 1 - self.test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                random_state=self.r_state,
                                                                                train_size=self.train_size,
                                                                                test_size=self.test_size)

    def print_options(self, y_pred):
        if self.is_print:
            np.set_printoptions(precision=self.print_option_precision)
            print(np.concatenate((y_pred.reshape(-1, 1), self.y_test.reshape(-1, 1)), 1))

    def confusion_matrix_info(self, y_pred):
        if self.is_confusion_matrix:
            print(confusion_matrix(self.y_test, y_pred))

    def encode_categorical_data(self):
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder=self.remainder)
        self.X = np.array(ct.fit_transform(self.X))

    def encode_label_dependent_variable(self):
        self.y = self.label_encoder.fit_transform(self.y)

    def __feature_scaling__(self):
        if self.std_scale:
            self.sc_X = StandardScaler()
            self.sc_y = StandardScaler()
            if self.feature_x and self.feature_y:
                self.X_train = self.sc_X.fit_transform(self.X_train)
                self.X_test = self.sc_X.transform(self.X_test)
                self.y_train = self.sc_y.fit_transform(self.y_train)
                self.y_test = self.sc_y.transform(self.y_test)
            elif self.feature_x:
                self.X_train = self.sc_X.fit_transform(self.X_train)
                self.X_test = self.sc_X.transform(self.X_test)
            elif self.feature_y:
                self.y_train = self.sc_y.fit_transform(self.y_train)
                self.y_test = self.sc_y.transform(self.y_test)
        else:
            self.n_X = Normalizer()
            self.n_y = Normalizer()
            if self.feature_x and self.feature_y:
                self.X_train = self.n_X.fit_transform(self.X_train)
                self.X_test = self.n_X.transform(self.X_test)
                self.y_train = self.n_y.fit_transform(self.y_train)
                self.y_test = self.n_y.transform(self.y_test)
            elif self.feature_x:
                self.X_train = self.n_X.fit_transform(self.X_train)
                self.X_test = self.n_X.transform(self.X_test)
            elif self.feature_y:
                self.y_train = self.n_y.fit_transform(self.y_train)
                self.y_test = self.n_y.transform(self.y_test)

    def multiple_linear_regression(self):
        self.lr_model.fit(self.X_train, self.y_train)
        y_pred = self.lr_model.predict(self.X_test)
        self.print_options(y_pred)
        score = r2_score(self.y_test, y_pred)*100
        self.regression_saiyan_model[score] = self.regression_saiyan_model.get(score, []) + [self.lr_model]

    def polynomial_regression(self):
        X_poly = self.pr_model.fit_transform(self.X_train)
        self.lr_model.fit(X_poly, self.y_train)
        y_pred = self.lr_model.predict(self.pr_model.transform(self.X_test))
        self.print_options(y_pred)
        score = r2_score(self.y_test, y_pred)*100
        self.regression_saiyan_model[score] = self.regression_saiyan_model.get(score, []) + [self.lr_model]

    def support_vector_regression(self):
        self.svr_model.fit(self.X_train, self.y_train)
        y_pred = self.svr_model.predict(self.X_test)
        self.print_options(y_pred)
        score = r2_score(self.y_test, y_pred)*100
        self.regression_saiyan_model[score] = self.regression_saiyan_model.get(score, []) + [self.svr_model]

    def decision_tree_regression(self):
        self.dtr_model.fit(self.X_train, self.y_train)
        y_pred = self.dtr_model.predict(self.X_test)
        self.print_options(y_pred)
        score = r2_score(self.y_test, y_pred)*100
        self.regression_saiyan_model[score] = self.regression_saiyan_model.get(score, []) + [self.dtr_model]

    def random_forest_regression(self):
        self.rf_model.fit(self.X_train, self.y_train)
        y_pred = self.rf_model.predict(self.X_test)
        self.print_options(y_pred)
        score = r2_score(self.y_test, y_pred)*100
        self.regression_saiyan_model[score] = self.regression_saiyan_model.get(score, []) + [self.rf_model]

    def logistic_regression(self):
        self.log_reg_model.fit(self.X_train, self.y_train)
        y_pred = self.log_reg_model.predict(self.X_test)
        self.print_options(y_pred)
        self.confusion_matrix_info(y_pred)
        score = accuracy_score(self.y_test, y_pred)*100
        self.regression_saiyan_model[score] = self.regression_saiyan_model.get(score, []) + [self.log_reg_model]

    def knn_classifier(self):
        self.knn_classifier_model.fit(self.X_train, self.y_train)
        y_pred = self.knn_classifier_model.predict(self.X_test)
        self.print_options(y_pred)
        self.confusion_matrix_info(y_pred)
        score = accuracy_score(self.y_test, y_pred)*100
        self.classification_saiyan_model[score] = self.classification_saiyan_model.get(score,
                                                                                       []) + [self.knn_classifier_model]

    def support_vector_classifier(self):
        self.svc_model.fit(self.X_train, self.y_train)
        y_pred = self.svc_model.predict(self.X_test)
        self.print_options(y_pred)
        self.confusion_matrix_info(y_pred)
        score = accuracy_score(self.y_test, y_pred)*100
        self.classification_saiyan_model[score] = self.classification_saiyan_model.get(score, []) + [self.svc_model]

    def naive_bayes(self):
        self.nb_model.fit(self.X_train, self.y_train)
        y_pred = self.nb_model.predict(self.X_test)
        self.print_options(y_pred)
        self.confusion_matrix_info(y_pred)
        score = accuracy_score(self.y_test, y_pred)*100
        self.classification_saiyan_model[score] = self.classification_saiyan_model.get(score, []) + [self.nb_model]

    def decision_tree_classifier(self):
        self.dtc_model.fit(self.X_train, self.y_train)
        y_pred = self.dtc_model.predict(self.X_test)
        self.print_options(y_pred)
        self.confusion_matrix_info(y_pred)
        score = accuracy_score(self.y_test, y_pred)*100
        self.classification_saiyan_model[score] = self.classification_saiyan_model.get(score, []) + [self.dtc_model]

    def random_forest_classifier(self):
        self.rfc_model.fit(self.X_train, self.y_train)
        y_pred = self.rfc_model.predict(self.X_test)
        self.print_options(y_pred)
        self.confusion_matrix_info(y_pred)
        score = accuracy_score(self.y_test, y_pred)*100
        self.classification_saiyan_model[score] = self.classification_saiyan_model.get(score, []) + [self.dtc_model]

    def transform_regression_to_saiyan(self):
        self.multiple_linear_regression()
        self.polynomial_regression()
        self.decision_tree_regression()
        self.random_forest_regression()
        self.support_vector_regression()
        print(self.regression_saiyan_model)

    def transform_classification_to_saiyan(self):
        self.logistic_regression()
        self.knn_classifier()
        self.support_vector_classifier()
        self.naive_bayes()
        self.decision_tree_classifier()
        self.random_forest_classifier()


if __name__ == "__main__":
    event_loop = asyncio.get_event_loop()
    start = datetime.datetime.now()
    # t = event_loop.run_until_complete(
    #     saiyan_model(file_name="High rainfall.csv").transform_regression_to_saiyan())
    saiyan_model(file_name="High rainfall.csv").transform_regression_to_saiyan()
    end = datetime.datetime.now()
    print((end - start))
