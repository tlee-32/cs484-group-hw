# Library imports
import smart_open
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
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
    sparseTrainingMatrix = sparsify(trainingRows)
    sparseTestMatrix = sparsify(testRows)
    
    # Define feature-selection estimators
    featureEstimator = RandomForestClassifier(n_estimators=100, n_jobs=2)
    featureSelectionModel = SelectFromModel(featureEstimator)

    # Initialize classifier pipeline with feature selection model
    classifierPipeline = Pipeline([('feature_selection', featureSelectionModel)])

    # Cross-validation that retrieves the optimal solver and penalty parameters
    #solverParam, penaltyParam = getOptimalCVParameters(classifierPipeline, sparseTrainingMatrix, labels)
    #print(solverParam)
    #print(penaltyParam)
    #return

    # Define main classifier
    logReg = LogisticRegression(C=1e6, solver='saga', penalty='l2', max_iter=15000, class_weight='balanced', multi_class='ovr', n_jobs=3)
    
    # Finalize and train classifier pipeline 
    classifierPipeline.steps.append(('classification', logReg))
    classifierPipeline.fit(sparseTrainingMatrix, labels)

    # Classify and write predictions to file
    classifyMoleculeActivity(classifierPipeline, sparseTestMatrix)

    print('\nMolecule activity successfully written to predictions.data (%d seconds)' % (time.time() - startTime))

"""
    Classifies the active/inactive molecules for test data
"""
def classifyMoleculeActivity(classifierPipeline, sparseTestMatrix):
    with smart_open.smart_open("./data/predicitions.data", "w") as f:
        predictions = classifierPipeline.predict(sparseTestMatrix)
        for prediction in predictions:
            if(prediction == 1):
                f.write("1\n")
            else:
                f.write("0\n")

"""
    Retrieves the penalty and solver with the highest average.
"""
def getOptimalCVParameters(classifierPipeline, sparseTrainingMatrix, labels):
    # liblinear, and sag ONLY support l1, while all 5 support l2
    solverMap = {'l1': ['liblinear', 'saga'], 
                 'l2': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'] }

    # Perform cross validation 
    solverScores = logRegCVScore(classifierPipeline, solverMap, sparseTrainingMatrix, labels)
    
    print("\nFinal scores: ")
    for tup in solverScores[::-1]:
        solverPenalty = tup[0][0] + " " + tup[0][1]
        score = tup[1]
        print(solverPenalty + ": \t%f" % score)

    optimalSolver = solverScores[-1][0][0]
    optimalPenalty = solverScores[-1][0][1]
    return optimalSolver, optimalPenalty
    
if __name__ == '__main__':
    main()
