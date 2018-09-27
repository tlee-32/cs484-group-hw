import smart_open
from preprocess.filetokenizer import readRows
from crossvalidation.crossvalidation import *
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, chi2
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

def main():
    trainingRows, labels = readRows("./data/train_drugs.data", loadFile=True, isTrainingFile=True)
    testRows, _ = readRows("./data/test_drugs.data", loadFile=True, isTrainingFile=False)
    lengths = [len(r) for r in trainingRows]
    maxLength = max(lengths)
    [r.extend([0]*(maxLength - len(r))) for r in trainingRows]
    """
        1) Convert filtered sparse matrix into original sparse matrix
        2) 

    """
    #lrcv = LogisticRegressionCV(Cs=10, dual=False, random_state=0, scoring='f1').fit(trainingRows,labels)
    #print(lrcv.scores_)
    #model = VarianceThreshold(threshold=(.8 * (1 - .8)))
    
    etc = ExtraTreesClassifier(n_estimators=50)
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(trainingRows, labels)
    model = SelectFromModel(etc)#SelectFromModel(lsvc)

    logReg = LogisticRegression(penalty = 'l1', random_state = 0)
    clf = Pipeline([
        ('feature_selection', model),
        ('classification', logReg)
    ])
    clf.fit(trainingRows, labels)
    print(logRegCrossValidation(clf, trainingRows, labels, 6, 'f1'))
    
    print('Molecule activity successfully written to predictions.data')
    
"""
    Classifies the active/inactive molecules for test data
"""
def classifyMoleculeActivity(classifier, testRows):
    with smart_open.smart_open("./data/predicitions.data", "w") as f:
        for row in testRows:
            label = None # TODO: replace with classifier
            if(label == '1'):
                f.write("1\n")
            else:
                f.write("0\n")

if __name__ == '__main__':
    main()