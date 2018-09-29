# Library imports
import smart_open
import time
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel 
from sklearn.pipeline import Pipeline
# Custom file imports
from preprocess.filetokenizer import readRows
from preprocess.sparsifier import sparsify
from crossvalidation.crossvalidation import *

def main():
    startTime = time.time()
    # Retrieve data
    trainingRows, labels = readRows("./data/train_drugs.data", loadFile=True, isTrainingFile=True)
    testRows, _ = readRows("./data/test_drugs.data", loadFile=True, isTrainingFile=False)
    
    # Convert data into csr matrix
    sMatrix = sparsify(trainingRows)
    
    # Define feature-selection estimators
    featureEstimator = ExtraTreesClassifier(n_estimators=50)
    featureSelectionModel = SelectFromModel(featureEstimator)

    # Initialize classifier pipeline with feature selection model
    classifierPipeline = Pipeline([('feature_selection', featureSelectionModel)])

    # newton-cg, lbfgs, and sag ONLY support l2, while all 5 support l1
    solverMap = {'l1': ['liblinear', 'saga'], 
                 'l2': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'] }

    # Perform cross validation 
    solverScores = logRegCVScore(classifierPipeline, solverMap, sMatrix, labels)
    
    print("\nFinal scores: ")
    for tup in solverScores[::-1]:
        print(tup[0] + ": \t%f" % tup[1])
    
    print('\nMolecule activity successfully written to predictions.data (%d seconds)' % (time.time() - startTime))
    
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
