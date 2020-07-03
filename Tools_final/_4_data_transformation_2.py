""" Step 4-2. Data transformation: predicting vessel tracks berthed (yes or no)
 Input: Terminal data frame with every vessel track attached with 10 different features
 Actions: Training multiple machine learning algorithms (final: XGBoost) in order to predict whether or not a vessel
 track has berthed at the terminal or not
 Output: Classifier (machine learning algorithm, best performed: XGBoost)
 """

import treeplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
import graphviz
import pandas as pd
# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance, plot_tree
import matplotlib.pyplot as plt
from _2_data_cleaning import drop_and_report
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


# Remove all vessel tracks with time in polygon < 60*30:
def remove_little_messages_tot(data):
    drop_list = list()
    for row in data.itertuples():
        if data.at[row.Index, 'time_in_polygon'] < 60*30:
            drop_list.append(row.Index)
            continue
    data = drop_and_report(data, drop_list, 'Remove obvious non-berthing tracks')
    return data


# Remove vessel tracks with no MMSI
def drop_mmsi_zero(data):
    drop_list = list()
    for row in data.itertuples():
        if len(str(row.mmsi)) < 9:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, 'Remove unused rows')
    return data


# Keep only vessel tracks that are wrongfully predicted
def data_wrongfully_predicted(data):
    drop_list = list()
    for row in data.itertuples():
        if row.track_berthed == row.y_pred:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, 'Remove correct predictions')
    return data


# Keep the false negative values (remove y_pred = 1)
def keep_fn(data):
    drop_list = list()
    for row in data.itertuples():
        if row.y_pred > 0:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, 'Keep only FN')
    return data


# Keep the false positive values (remove y_pred = 0)
def keep_fp(data):
    drop_list = list()
    for row in data.itertuples():
        if row.y_pred < 1:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, 'Keep only FP')
    return data


# Keep only vessel tracks that are correctly predicted
def data_correctly_predicted(data):
    drop_list = list()
    for row in data.itertuples():
        if row.track_berthed != row.y_pred:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, 'Remove correct predictions')
    return data


# Keep the true negative values (remove y_pred = 1)
def keep_tn(data):
    drop_list = list()
    for row in data.itertuples():
        if row.y_pred > 0:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, 'Keep only TN')
    return data


# Keep the true positive values (remove y_pred = 0)
def keep_tp(data):
    drop_list = list()
    for row in data.itertuples():
        if row.y_pred < 1:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, 'Keep only TP')
    return data


# Test handle
if __name__ == '__main__':
    import pickle
    """ Start after all cleaning and preprocessing steps have been performed  """
    # Container terminals
    data_processed_Rdam = pd.read_csv('Data-frames/Old_results_during_phase2/Features_ct_rdam_euromax.csv')
    data_processed_APMRdam = pd.read_csv('Data-frames/Old_results_during_phase2/Features_ct_rdam_apm.csv')
    data_processed_BEST = pd.read_csv('Data-frames/Old_results_during_phase2/Features_ct_best.csv')
    data_processed_Lisbon = pd.read_csv('Data-frames/Old_results_during_phase2/Features_ct_lisbon.csv')

    # Dry Bulk Terminals
    data_processed_RdamEMO = pd.read_csv('Data-frames/Old_results_during_phase2/Features_db_rdam.csv')
    data_processed_Vliss = pd.read_csv('Data-frames/Old_results_during_phase2/Features_db_vliss.csv')
    data_processed_LisDB = pd.read_csv('Data-frames/Old_results_during_phase2/Features_db_lisbon.csv')
    data_processed_NHDB = pd.read_csv('Data-frames/Old_results_during_phase2/Features_db_NH.csv')

    # Liquid Bulk Terminals
    data_processed_RdamLBT = pd.read_csv('Data-frames/Old_results_during_phase2/Features_lb_rdam.csv')
    data_processed_Vliss_LBT = pd.read_csv('Data-frames/Old_results_during_phase2/Features_lb_vliss.csv')
    data_processed_Lisbon_LBT = pd.read_csv('Data-frames/Old_results_during_phase2/Features_lb_lisbon.csv')
    data_processed_belfast_LBT = pd.read_csv('Data-frames/Old_results_during_phase2/Features_lb_belfast.csv')

    # Merge all data (Container Terminals)
    data_CT_1 = pd.concat([data_processed_Rdam, data_processed_APMRdam, data_processed_BEST, data_processed_Lisbon
                           ], ignore_index=True)
    data_CT = drop_mmsi_zero(data_CT_1)

    # Merge all necessary data (Dry Bulk Terminals)
    data_DBT_1 = pd.concat([data_processed_RdamEMO, data_processed_Vliss, data_processed_LisDB
                               , data_processed_NHDB
                            ], ignore_index=True)
    data_DBT = drop_mmsi_zero(data_DBT_1)

    # Merge all necessary data (Liquid Bulk Terminals)
    data_LBT_1 = pd.concat([data_processed_RdamLBT, data_processed_Vliss_LBT, data_processed_Lisbon_LBT
                               , data_processed_belfast_LBT], ignore_index=True)
    data_LBT = drop_mmsi_zero(data_LBT_1)

    # Remove 'clear' not berthed vessel tracks
    remove_little_messages_tot(data_CT)
    remove_little_messages_tot(data_DBT)
    remove_little_messages_tot(data_LBT)

    feature_cols = ['avg_timestamp_interval', 'messages_tot', 'loa', 'mean_75_sog_per_track', 'DWT',
                    'message_frequency', 'distance_avg', 'mean_sog_per_track', 'std_sogms', 'std_distance',
                    'time_in_polygon',
                    'teu_capacity', 'std_location']

    X_CT = data_CT[feature_cols]  # Features
    y_CT = data_CT.track_berthed  # Target variable

    X_DBT = data_DBT[feature_cols]  # Features
    y_DBT = data_DBT.track_berthed  # Target variable

    X_LBT = data_LBT[feature_cols]  # Features
    y_LBT = data_LBT.track_berthed  # Target variable

    # Merge terminal types
    X = pd.concat([X_CT, X_DBT, X_LBT], ignore_index=True)
    y = pd.concat([y_CT, y_DBT, y_LBT], ignore_index=True)

    # Fill all NaN by 0
    X_2 = X.fillna(0)

    # Split data set into training set and test set (80% training and 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size=0.2, random_state=2)


    # """ 5. Decision Tree """
    # # Feature scaling: input variables to same range (dependent on type of model): not necessary
    # # Fitting to training set
    # from sklearn.tree import DecisionTreeClassifier
    # classifier = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=0)
    # # most common = entropy (information gain): quality of a split
    # # gini, better to understand
    # classifier.fit(X_train, y_train)
    # # Predicting test set results
    # y_pred = classifier.predict(X_test)
    # # Making the confusion Matrix
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred)
    # # Model Accuracy, how often is the classifier correct?
    # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


    # define custom class to fix bug in xgboost 1.0.2
    class MyXGBClassifier(XGBClassifier):
        @property
        def coef_(self):
            return None

    """ 7. XGBOOST """
    # Feature scaling: input variables to same range (dependent on type of model): not necessary
    # Fitting to training set
    from xgboost import XGBClassifier
    classifier = MyXGBClassifier(max_depth=4, random_state=0, learning_rate=0.2, n_estimators=100)
    classifier.fit(X_train, y_train)
    # Predicting test set results
    y_pred = classifier.predict(X_test)
    # Making the confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    # Model Accuracy, how often is the classifier correct?
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    print('Mean of 10 accuracies is: ', accuracies.mean(), 'and std: ', accuracies.std())
    # print(accuracy_score(y_test, y_pred))

    """ Save trained model """
    with open('classifier_pickle', 'wb') as f:
        pickle.dump(classifier, f)



    # eval_set = [(X_train, y_train), (X_test, y_test)]
    # eval_metric = ["auc", "error"]
    # classifier.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

    # # Plot the tree
    # dot_data = tree.export_graphviz(classifier, out_file=None, feature_names=feature_cols,
    #                                 class_names=['not_berthed', 'berthed'], filled=True, rounded=True,
    #                                 special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph.render("iris")

    # # List of values to try for max_depth:
    # max_depth_range = list(range(1, 7))  # List to store the average RMSE for each value of max_depth:
    # accuracy = []
    # for depth in max_depth_range:
    #     # clf = DecisionTreeClassifier(max_depth=depth, criterion='gini',
    #     #                              random_state=0)
    #     clf = XGBClassifier(max_depth=depth, warm_start,  random_state=0, n_estimators=100)
    #     clf.fit(X_train, y_train)
    #     score = clf.score(X_test, y_test)
    #     accuracy.append(score)
    # plt.plot(max_depth_range, accuracy)
    # plt.xlabel('Max_depth')
    # plt.ylabel('Accuracy score (%)')
    # plt.title('Find optimum number of layers in decision tree')

    # import numpy as np
    # importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(classifier.feature_importances_,  3)})
    # importances = importances.sort_values('importance', ascending=False)

    # #     Visualisation
#    for XGBoost
# import shap
# explainer = shap.TreeExplainer(classifier)
# shap_values = explainer.shap_values(X_train)
# #shap.summary_plot(shap_values, X_train, plot_type='bar')
# shap.TreeExplainer(classifier).shap_interaction_values(X_train)
# shap.summary_plot(shap_values, X_train)


# Plot confusion matrix
    # from mlxtend.plotting import plot_confusion_matrix
    # import matplotlib.pyplot as plt
    # import numpy as np
    # fig, ax = plot_confusion_matrix(conf_mat=cm)
    # plt.title('Confusion matrix')
    # plt.show()

    # # Print accuracies for berthing only
    # import numpy as np
    # cm_fl = cm.flatten()
    # TN = cm_fl[0]
    # FP = cm_fl[1]
    # FN = cm_fl[2]
    # TP = cm_fl[3]
    # print('Total accuracy is', np.round((TN+TP)/(TN+TP+FP+FN),5)*100, '%')
    # print('Percentage of correctly predicted berths (compared to total number of actual berths', np.round((TP)/(FN+TP),
    #                                                                                                       5)*100, '%')
    # print('False number berths compared to total berths', np.round(FP/(FN+TP)*100,5))
    #
    # """ Inspect wrongful predictions """
    # #
    # # Merge terminal types
    # X_all = pd.concat([data_CT, data_DBT, data_LBT], ignore_index=True)
    # # Fill all NaN by 0
    # X_3 = X_all.fillna(0)
    # # Predict all to investigate where it goes wrong
    # y_pred_all = classifier.predict(X_2)
    # # Making the confusion Matrix
    # cm = confusion_matrix(y, y_pred_all)
    # # # Model Accuracy, how often is the classifier correct?
    # # accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    # # print('Mean of 10 accuracies is: ', accuracies.mean(), 'and std: ', accuracies.std())
    #
    # # Attach predictions to old data set
    # y_pred = pd.DataFrame(y_pred_all, columns=['y_pred'])
    # df_all = (pd.concat([X_3, y_pred], axis=1))
    # df_all.to_csv('df_all.csv')
    #
    # # # Inspect wrongfully predicted values (append dataframes with actual y and predicted y)
    # # df = (pd.concat([X_test, y_test], axis=1, ignore_index=False)).reset_index()
    # # ypred = pd.DataFrame(y_pred, columns=['y_pred'])
    # # df_merged = pd.concat([df, ypred], axis=1)
    # df_all_1 = df_all.copy()
    # df_all_2 = df_all.copy()
    #
    # # Keep only vessel tracks that are wrongfully predicted
    # df = data_wrongfully_predicted(df_all_1)
    # # For wrong predictions
    # df_1 = df.copy()
    # df_2 = df.copy()
    #
    # df_FN = keep_fn(df_1)
    # df_FP = keep_fp(df_2)
    #
    # # For correct predictions
    # # Keep only vessel tracks that are wrongfully predicted
    # df_corr = data_correctly_predicted(df_all_2)
    # df_3 = df_corr.copy()
    # df_4 = df_corr.copy()
    #
    # df_TN = keep_tn(df_3)
    # df_TP = keep_tp(df_4)
    #
    # df_FN.to_csv('FN.csv')
    # df_FP.to_csv('FP.csv')
    # df_TN.to_csv('TN.csv')
    # df_TP.to_csv('TP.csv')





