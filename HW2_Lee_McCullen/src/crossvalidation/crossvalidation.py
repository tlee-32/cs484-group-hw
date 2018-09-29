import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

"""
    Perform cross-validation on given selection of l1 and l2 solvers.

    return - f1 scores of each solver
"""
def logRegCVScore(classifierPipeline, solverMap, sMatrix, labels):
    scores = []
    """
        Test each type of solver (numerical optimizers) with its associated penalty.
        sag and saga tend to have convergence warnings (tested up to max_iter=5000 (default is 100))
    """
    for penalty, solvers in solverMap.items():
        for s in solvers:
            print("\nSOLVER: " + penalty + " " + s)
            logReg = LogisticRegression(solver=s, penalty=penalty, max_iter=10000, multi_class='ovr')

            # Add classification to pipeline and train it
            classifierPipeline.steps.append(['classification', logReg])
            classifierPipeline.fit(sMatrix, labels)
            
            # Compute cross-validation scores
            avg = computeCVAverageScore(classifierPipeline, sMatrix, labels)
            
            print("avg score from 5x2 cv: %f\n" % avg)
            scores.append((penalty + " " + s, avg))

            # Reset current logReg classifier for the next solver
            classifierPipeline.steps.pop() 
            #print(clf.score(sMatrix, labels)) #<- seems to always return 1.0, may have to do with UndefinedMetricWarning
    scores = sorted(scores, key=lambda x: x[1]) # rank from best to worst solvers
    return scores

"""
    sets x folds (default: 5x2) cross validation.
    Default parameters compute the average of 5 sets of 2-fold CVs.

    return - overall average of total set averages
"""
def computeCVAverageScore(classifierPipeline, sMatrix, labels, sets=5, folds=2, scoring='f1'):
    avgs = []
    for _ in range(5):
        scores = cross_val_score(classifierPipeline, sMatrix, labels, cv=2, scoring='f1')
        avgs.append( np.average(scores) )
    print("avgs: " + str(avgs))
    return np.average(avgs)
