#Five different algorithms.

def gaussNB(features_train,features_test,labels_train,labels_test):

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    #alg_report("GaussianNB",labels_test,pred)

    return clf

def alg_report(alg,labels,pred):

    from sklearn.metrics import classification_report
    target_names = ["Not POI", "POI"]
    print (alg, " Classification Report:")
    print (classification_report(y_true=labels, y_pred=pred, target_names=target_names))

def DTree(features_train,features_test,labels_train,labels_test):

    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split=100)
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    #alg_report("Decision Tree",labels_test,pred)

    return clf

def LogReg(features_train,features_test,labels_train,labels_test):

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    #alg_report("Logistic Regression",labels_test,pred)

    return clf

def LinearS(features_train,features_test,labels_train,labels_test):

    from sklearn.svm import LinearSVC
    clf = LinearSVC(C=1.0)
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    #alg_report("Linear SVC",labels_test,pred)

    return clf

def RandForest(features_train,features_test,labels_train,labels_test):

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, min_samples_split=6, max_features=None,oob_score=True)
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    #alg_report("Random Forest",labels_test,pred)

    return clf
