#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from explore_data import drawboxes,printresults,printhighnonPOIs,checktotals,graphstats,newfeaturelist,finlistall,emaillist,update_data_errors,updatenulls
from feature_selection import new_features,test_kbest
from algorithms import gaussNB,LogReg,RandForest,LinearS, DTree

### feature_list and old_feature_list are lists of strings, each of which is a feature name.
### The first feature must always be "poi".

feature_list = ['poi','odd_payments','key_payments',
'retention_incentives','salary','bonus','total_stock_value',
'exercised_stock_options','total_of_totals','deferral_payments']

old_feature_list = ['poi','retention_incentives/key_payments','bonus/salary','poi_emailratio','exercised_stock_options/salary','odd_payments','key_payments','deferral_balance',
'retention_incentives','total_of_totals','salary','bonus','other','deferral_payments',
'deferred_income','loan_advances','long_term_incentive','director_fees','expenses','total_payments','total_stock_value',
'exercised_stock_options','restricted_stock_deferred','restricted_stock']

### Load the dictionary containing the dataset


with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

    #Task 1 and 2a: Initial exploration and Null value and Outlier analysis (functions in explore_data.py)

    printresults(data_dict)
    updatenulls(data_dict)
    
    #can run printresults() again to check if this has worked - is does!

    #printing out nonPOIs with high values
    printhighnonPOIs(data_dict)
    checktotals(data_dict)


    #Task 2b: Outlier removal (functions in explore_data.py)
    data_dict.pop('TOTAL')
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
    data_dict.pop('LOCKHART EUGENE E') #all null values.  Cheating as the reviewer told me about this one.
    update_data_errors(data_dict)

    ### Task 3: New features (functions in explore_data.py)

    #look at data graphically
    finstats,poi_finstats = graphstats(data_dict,finlistall)
    drawboxes(finstats,poi_finstats)

    data_dict = new_features(data_dict)
    #draw graphs for new features.
    finstats,poi_finstats = graphstats(data_dict,newfeaturelist)
    drawboxes(finstats,poi_finstats)

    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

### Task 4: Feature Removal and Selection ( functions in feature_selection.py)

#initial test commented out
#data = featureFormat(my_dataset, feature_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)
#test_kbest(features, labels,feature_list)

skb = SelectKBest(k=9)

### Task 5: Feature scaling

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler

scaler =MinMaxScaler()

### Task 8 - Algorithm Choice and Parameter Tuning (functions in algorithms.py)
#uncomment the function calls below to test on each algorithm

#clf = gaussNB(scaler,skb)
clf = DTree(scaler,skb)
#clf = LogReg(scaler,skb)
#clf = LinearS(scaler,skb)
#clf = RandForest(scaler,skb)


#Task 7: Cross Validation and Testing - using StratifiedShuffleSplit and report in tester.py (Udacity function)
test_classifier(clf,my_dataset,feature_list)

### Task 6: Dump your classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, feature_list)
