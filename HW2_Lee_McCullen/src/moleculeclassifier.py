import smart_open
from preprocess.filetokenizer import readRows
from crossvalidation.crossvalidation import *
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, chi2
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt

def main():
    trainingRows, labels = readRows("./data/train_drugs.data", loadFile=True, isTrainingFile=True)
    testRows, _ = readRows("./data/test_drugs.data", loadFile=True, isTrainingFile=False)
    lengths = [len(r) for r in trainingRows]
    maxLength = max(lengths)
    #[r.extend([0]*(maxLength - len(r))) for r in trainingRows]
    """
        1) Convert filtered sparse matrix into original sparse matrix
        2) 

    """
    
    csrTup = ([], [])
    csrData = []
    for index, row in enumerate(trainingRows):
        for idxToken in row:
            csrTup[0].append(index)
            csrTup[1].append(int(idxToken))
            csrData.append(1)
    
    sMatrix = csr_matrix((csrData, csrTup))

    etc = ExtraTreesClassifier(n_estimators=50)
    #lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(sMatrix, labels)
    #lrcv = LogisticRegressionCV(Cs=10, dual=False, random_state=0, scoring='f1').fit(trainingRows,labels)
    #print(lrcv.scores_)

    #model = VarianceThreshold(threshold=(.8 * (1 - .8)))
    model = SelectFromModel(etc)#SelectFromModel(lsvc)

    """
        Test each type of solver.
        sag and saga tend to have convergence warnings (tested up to max_iter=5000 (default is 100))
    """
    solverScores = []
    for s in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
        for l in ['l1', 'l2']:
            #newton-cg, lbfgs, and sag ONLY support l2, while all 5 support l1
            if ( (l == 'l1' and (s == 'liblinear' or s == 'saga')) or l == 'l2'):
                logReg = LogisticRegression(solver=s, penalty=l, max_iter=10000, multi_class='ovr')
                #logReg = LogisticRegression(solver='liblinear', penalty = 'l2')
                #logReg = LogisticRegressionCV(solver='liblinear', penalty='l1', Cs=10, cv=2, dual=False, random_state=0, scoring='f1')
                clf = Pipeline([
                    ('feature_selection', model),
                    ('classification', logReg)
                ])
                clf.fit(sMatrix, labels)

                """
                    5x2 fold cross validation: avg of 5 sets of 2-fold CVs
                """
                avgs = []
                #for _ in range(5):
                #    scores = logRegCrossValidation(clf, sMatrix, labels, 2, 'f1')
                #    avgs.append( np.average(scores) )

                for _ in range(5):
                    scores = logRegCrossValidation(clf, sMatrix, labels, 2, 'f1')
                    avgs.append( np.average(scores) )

                avg = np.average(avgs)
                print("\nSOLVER: " + l + " " + s)
                print("avgs: " + str(avgs))
                print("avg score from 5x2 cv: %f\n" % avg)
                solverScores.append((l + " " + s, avg))

                #print(clf.score(sMatrix, labels)) #<- seems to always return 1.0, may have to do with UndefinedMetricWarning
    solverScores = sorted(solverScores, key=lambda x: x[1])
    print("\nFinal scores: ")
    for tup in solverScores[::-1]:
        print(tup[0] + ": \t%f" % tup[1])
    
    print('\nMolecule activity successfully written to predictions.data')
    
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
