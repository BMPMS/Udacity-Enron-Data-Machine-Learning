from sklearn.pipeline import Pipeline
#Five different algorithms all using piplines

def gaussNB(scaler,skb):

    from sklearn.naive_bayes import GaussianNB

    gnb = GaussianNB()
    clf = Pipeline(steps=[('Scaler',scaler), ('SKB', skb), ('NaiveBayes', gnb)])

    return clf


def DTree(scaler,skb):

    from sklearn import tree
    dt = tree.DecisionTreeClassifier(random_state=42,min_samples_split=5,splitter='random')
    clf = Pipeline(steps=[('Scaler',scaler), ('SKB', skb), ('Decision Tree', dt)])

    return clf


def LogReg(scaler,skb):

    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression(class_weight='balanced')
    clf = Pipeline(steps=[('Scaler',scaler), ('SKB', skb), ('Logistic Regression', lg)])

    return clf


def LinearS(scaler,skb):

    from sklearn.svm import LinearSVC

    lsvc = LinearSVC(C=10,class_weight='balanced')
    clf = Pipeline(steps=[('Scaler',scaler), ('SKB', skb), ('Linear SVC', lsvc)])


    return clf

def RandForest(scaler,skb):

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100,bootstrap=False)
    clf = Pipeline(steps=[('Scaler',scaler), ('SKB', skb), ('Random Forest', rf)])

    return clf
