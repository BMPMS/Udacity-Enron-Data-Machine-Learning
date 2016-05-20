def kbestperc(features_train,features_test,labels_train,labels_test,features_list):

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import SelectPercentile

    kbest = SelectKBest(k=4)
    train_new = kbest.fit_transform(features_train,labels_train)
    kbest2 = SelectKBest(k=4)
    test_new = kbest2.fit_transform(features_test,labels_test)
    x = 0

    perc = SelectPercentile(percentile=60)
    train_perc = perc.fit_transform(features_train,labels_train)
    perc2 = SelectPercentile(percentile=60)
    test_perc = perc2.fit_transform(features_test,labels_test)

    for f in features_list:
        if x > 0:
            print(f,'train?',kbest.scores_[x-1],'test?',kbest2.scores_[x-1])
            print(f,'train%',perc.scores_[x-1],'test%',perc2.scores_[x-1])

        x = x + 1

    print('training keep:',kbest.get_support())
    print('test scores:',kbest2.get_support())
    print('training keep:',perc.get_support())
    print('test scores:',perc2.get_support())
