#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from explore_data import drawboxes,printresults,printhighnonPOIs,checktotals,graphstats,newfeaturelist,finlistall,emaillist,update_data_errors,updatenulls
from feature_selection import new_features,test_kbest
from algorithms import gaussNB,LogReg,RandForest,LinearS, DTree

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

feature_list = ['poi','bonus/salary','exercised_stock_options/salary','poi_emailratio','exercised_stock_options/salary','odd_payments','key_payments','deferral_balance',
'retention_incentives','total_of_totals','salary','bonus','other','deferral_payments',
'deferred_income','loan_advances','long_term_incentive',
'exercised_stock_options','restricted_stock_deferred','restricted_stock']

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


    #Task 2: Remove outliers..
    data_dict.pop('TOTAL')
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
    update_data_errors(data_dict)

    ### Task 3: Create new feature(s)

    #look at data graphically
    finstats,poi_finstats = graphstats(data_dict,finlistall)
    drawboxes(finstats,poi_finstats)

    data_dict = new_features(data_dict)
    #draw graphs for new features.
    finstats,poi_finstats = graphstats(data_dict,newfeaturelist)
    drawboxes(finstats,poi_finstats)

    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

### Extract features and labels from dataset for local testing
#In the end I decided to use the StratifiedShuffleSplit in the testing function to do this


### optimise my features with KBest
#initial test of kbest (see document)
#test_kbest(features, labels,feature_list)

### Task 4: Try a varity of classifiers (5 algorithms - see algorithms.py)

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler

scaler =MinMaxScaler()
skb = SelectKBest(k=6)


#uncomment the function calls below to test on each algorithm

#clf = gaussNB(scaler,skb)
clf = DTree(scaler,skb)
#clf = LogReg(scaler,skb)
#clf = LinearS(scaler,skb)
#clf = RandForest(scaler,skb)


#Task 5: Validation and Testing - using StratifiedShuffleSplit and report in tester.py
test_classifier(clf,my_dataset,feature_list)

### Task 6: Dump your classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, feature_list)
