#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from explore_data import printresults,printhighnonPOIs,checktotals,graphstats,drawgraphs,finlistall,emaillist,poiemails,update_data_errors,updatenulls
from feature_selection import kbestperc
from algorithms import gaussNB,LogReg,RandForest,LinearS, DTree

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

feature_list = ['poi','poi_emails','key_payments','deferral_balance',
'retention_incentives','total_of_totals','loan_advances']

### Load the dictionary containing the dataset


with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

    #Task 1: Explore data (functions in explore_data.py)

    printresults(data_dict)
    updatenulls(data_dict)
    #can run printresults() again to check if this has worked - is does!

    #printing out nonPOIs with high values
    printhighnonPOIs(data_dict)
    checktotals(data_dict)


    #check total payments and stock values
    checktotals(data_dict)

    #Task 2: Remove outliers..
    data_dict.pop('TOTAL')
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
    update_data_errors(data_dict)

    #gather graphstats and drawgraphs
    finstats,poi_finstats = graphstats(data_dict,finlistall)
    emailstats,poi_emailstats = graphstats(data_dict,emaillist)
    drawgraphs(finstats,poi_finstats,emailstats)

### Task 3: Create new feature(s)
    for items in data_dict:
        data_dict[items]['key_payments'] = data_dict[items]['salary'] + data_dict[items]['bonus'] + data_dict[items]['other']
        data_dict[items]['deferral_balance'] = data_dict[items]['deferral_payments'] + data_dict[items]['deferred_income']
        data_dict[items]['retention_incentives'] = data_dict[items]['long_term_incentive'] + data_dict[items]['total_stock_value']
        data_dict[items]['total_of_totals'] = data_dict[items]['total_payments'] + data_dict[items]['total_stock_value']
        data_dict[items]['poi_emails'] = poiemails(data_dict[items])
    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### fit the classifier using training set, and test on test set
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

#Kbest and Percentile attempted but no consistent results so chose features manually
#kbestperc(features_train, features_test, labels_train, labels_test,features_list)

#scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.fit_transform(features_test)


### Task 4: Try a varity of classifiers (5 algorithms)

#clf = LinearS(features_train,features_test,labels_train,labels_test)
#clf = gaussNB(features_train,features_test,labels_train,labels_test)
#clf = LogReg(features_train,features_test,labels_train,labels_test)
#clf = DTree(features_train,features_test,labels_train,labels_test)
#CHOSEN ALGORITHM - Random Forest classifier
clf = RandForest(features_train,features_test,labels_train,labels_test)

test_classifier(clf,my_dataset,feature_list)

### Task 6: Dump your classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, feature_list)
