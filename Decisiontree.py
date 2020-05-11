import treeplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
import graphviz
import pandas as pd
# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance, plot_tree
import matplotlib.pyplot as plt
from data_cleaning import drop_and_report


# Remove all vessel tracks with messages tot < 5/ time in polygon < 60*30:
def remove_little_messages_tot(data):
    drop_list = list()
    for row in data.itertuples():
        if data.at[row.Index, 'messages_tot'] < 5 or data.at[row.Index, 'time_in_polygon'] < 60*30:
            drop_list.append(row.Index)
            continue
    data = drop_and_report(data, drop_list, 'Remove obvious non-berthing tracks')
    return data


""" Start after all cleaning and preprocessing steps have been performed  """
# Container terminals
data_processed_SH = pd.read_csv('Data_forDT-SH-CT-15042019-01022020.csv')
data_processed_Rdam = pd.read_csv('Data_forDT-euromaxCT-Rdam-15042019-01022020.csv')
data_processed_APMRdam = pd.read_csv('Data_forDT-CT-RdamAPM-15042019-01022020.csv')
data_processed_BEST = pd.read_csv('Data_forDT-CT-BEST-01072019-01022020.csv')
data_processed_Lisbon = pd.read_csv('Data_forDT-CT-Lisbon-01072019-01022020.csv')
data_processed_Sing = pd.read_csv('Data_forDT-CT-Sing-01072019-01022020.csv')

# Dry Bulk Terminals
data_processed_RdamEMO = pd.read_csv('Data_forDT_RdamEMO-DB-01072019-01022020.csv')
data_processed_Vliss = pd.read_csv('Data_forDT_VlissDBT-01072019-01022020.csv')
data_processed_LisDB = pd.read_csv('Data_forDT-Lisbon_DBT-01072019-01022020.csv')
data_processed_NHDB = pd.read_csv('Data_forDT-NH_DBT-01072019-01022020.csv')
data_processed_Sing_DBT = pd.read_csv('Data_forDT_Sing_LBT_01072019-01022020.csv')

# Liquid Bulk Terminals
data_processed_RdamLBT = pd.read_csv('Data_forDT_Rdam_LBT_01072019-01022020.csv')
data_processed_Vliss_LBT = pd.read_csv('Data_forDT_Vliss_LBT_01072019-01022020.csv')
data_processed_Lisbon_LBT = pd.read_csv('Data_forDT_Lisbon_LBT_01072019-01022020.csv')
data_processed_Sing_LBT = pd.read_csv('Data_forDT_Sing_LBT_01072019-01022020.csv')
data_processed_UK_LBT = pd.read_csv('Data_forDT_UK_LBT-01072019-01022020.csv')

# Merge all data (Container Terminals)
data_CT = pd.concat([data_processed_SH, data_processed_Rdam, data_processed_APMRdam, data_processed_BEST,
                     data_processed_Lisbon, data_processed_Sing], ignore_index=True)

# Merge all necessary data (Dry Bulk Terminals)
data_DBT = pd.concat([data_processed_RdamEMO, data_processed_Vliss, data_processed_LisDB, data_processed_NHDB,
                      data_processed_Sing_DBT], ignore_index=True)

# Merge all necessary data (Liquid Bulk Terminals)
data_LBT = pd.concat([data_processed_RdamLBT, data_processed_Vliss_LBT, data_processed_Lisbon_LBT,
                      data_processed_Sing_LBT, data_processed_UK_LBT], ignore_index=True)

# Remove 'clear' unnecessary vessels (tot_messages<5)
remove_little_messages_tot(data_CT)
remove_little_messages_tot(data_DBT)
remove_little_messages_tot(data_LBT)

feature_cols = ['mean_sog_per_track', 'time_in_polygon', 'avg_timestamp_interval', 'messages_tot', 'distance_avg',
                'mean_75_sog_per_track', 'std_sogms', 'std_distance', 'message_frequency']

X_CT = data_CT[feature_cols]  # Features
y_CT = data_CT.track_berthed  # Target variable

X_DBT = data_DBT[feature_cols]  # Features
y_DBT = data_DBT.track_berthed  # Target variable

X_LBT = data_LBT[feature_cols]  # Features
y_LBT = data_LBT.track_berthed  # Target variable

# Merge terminal types
X = pd.concat([X_CT, X_DBT, X_LBT], ignore_index=True)
y = pd.concat([y_CT, y_DBT, y_LBT], ignore_index=True)


# Split dataset into training set and test set (80% training and 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# """ Fit chosen classifier to continue in next script """
# classifier = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
# classifier.fit(X, y)


# Test handle
if __name__ == '__main__':
    """  .......................... Different Machine learning models  .......................... """
    """ 1. Logistic Regression """
    # Feature scaling: input variables to same range (dependent on type of model)
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    # Fitting logistic regression to training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)  #random state to keep same results
    classifier.fit(X_train, y_train)
    # Predicting test set results
    y_pred = classifier.predict(X_test)
    # Making the confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))



    """ 2. K-nearest neighbours """
    # Feature scaling: input variables to same range (dependent on type of model)
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    # Fitting to training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean', p=2)
    classifier.fit(X_train, y_train)
    # Predicting test set results
    y_pred = classifier.predict(X_test)
    # Making the confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    """ 3. Support vector machine """
    # Feature scaling: input variables to same range (dependent on type of model)
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    # Fitting to training set
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf', random_state=0)  #different kernels: basic: 'linear': will look like logistic kernel
    # kernel: default = 'rbf'
    classifier.fit(X_train, y_train)
    # Predicting test set results
    y_pred = classifier.predict(X_test)
    # Making the confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    """ 4. Naive Bayes """
    # Feature scaling: input variables to same range (dependent on type of model)
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    # Fitting to training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    # Predicting test set results
    y_pred = classifier.predict(X_test)
    # Making the confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    """ 5. Decision Tree """
    # Feature scaling: input variables to same range (dependent on type of model): not necessary
    # Fitting to training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=0)
    # most common = entropy (information gain): quality of a split
    # gini, better to understand
    classifier.fit(X_train, y_train)
    # Predicting test set results
    y_pred = classifier.predict(X_test)
    # Making the confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # """ 6. Random Forest """
    # # Feature scaling: input variables to same range (dependent on type of model): not necessary
    # # Fitting to training set
    # from sklearn.ensemble import RandomForestClassifier
    # classifier = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=6, random_state=0)
    # # be aware of overfitting
    # classifier.fit(X_train, y_train)
    # # Predicting test set results
    # y_pred = classifier.predict(X_test)
    # # Making the confusion Matrix
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred)
    # # Model Accuracy, how often is the classifier correct?
    # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    #

    # """ 7. XGBOOST """
    # # Feature scaling: input variables to same range (dependent on type of model): not necessary
    # # Fitting to training set
    # from xgboost import XGBClassifier
    # classifier = XGBClassifier(max_depth=6, random_state=0, n_estimators=100)
    # classifier.fit(X_train, y_train)
    # # Predicting test set results
    # y_pred = classifier.predict(X_test)
    # # Making the confusion Matrix
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred)
    # # Model Accuracy, how often is the classifier correct?
    # from sklearn.model_selection import cross_val_score
    # accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    # print('Mean of 10 accuracies is: ', accuracies.mean(), 'and std: ', accuracies.std())

    # Plot the tree
    dot_data = tree.export_graphviz(classifier, out_file=None, feature_names=feature_cols,
                                    class_names=['not_berthed', 'berthed'], filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("iris")

    # List of values to try for max_depth:
    # max_depth_range = list(range(1, 7))  # List to store the average RMSE for each value of max_depth:
    # accuracy = []
    # for depth in max_depth_range:
    #     clf = DecisionTreeClassifier(max_depth=depth, criterion='gini',
    #                                  random_state=0)
    #     clf.fit(X_train, y_train)
    #     score = clf.score(X_test, y_test)
    #     accuracy.append(score)
    # plt.plot(max_depth_range, accuracy)
    # plt.xlabel('Max_depth')
    # plt.ylabel('Accuracy score (%)')
    # plt.title('Find optimum number of layers in decision tree')
    #
    # import numpy as np
    # importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(classifier.feature_importances_,  3)})
    # importances = importances.sort_values('importance', ascending=False)
    #
    # # Plot confusion matrix
    # from mlxtend.plotting import plot_confusion_matrix
    # import matplotlib.pyplot as plt
    # import numpy as np
    # fig, ax = plot_confusion_matrix(conf_mat=cm)
    # plt.title('Confusion matrix')
    # plt.show()
    #
    # # Visualisation for XGBoost
    # import shap
    # explainer = shap.TreeExplainer(clf)
    # shap_values = explainer.shap_values(X_train)
    #
    # #shap.summary_plot(shap_values, X_train, plot_type='bar')
    # shap.TreeExplainer(clf).shap_interaction_values(X_train)
    # shap.summary_plot(shap_values, X_train)
    # shap.dependence_plot('message_frequency', shap_values, X_train)

    # Print accuracies for berthing only
    import numpy as np
    cm_fl = cm.flatten()
    TN = cm_fl[0]
    FP = cm_fl[1]
    FN = cm_fl[2]
    TP = cm_fl[3]
    print('Total accuracy is', np.round((TN+TP)/(TN+TP+FP+FN),5)*100, '%')
    print('Percentage of correctly predicted berths (compared to total number of actual berths', np.round((TP)/(FN+TP),
                                                                                                          5)*100, '%')
    print('False number berths compared to total berths', np.round(FP/(FN+TP)*100,5))

