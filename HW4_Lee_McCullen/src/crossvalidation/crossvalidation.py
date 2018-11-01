import numpy as np
from surprise import accuracy
from surprise.model_selection import KFold

"""
    Perform k-fold cross-validation for the given classifier and
    calculate the Root Mean Square Error.

    return - average RMSE score across k-folds
"""
def computeCVAverageRMSE(trainData, predictionAlgorithm, k=5):
  avgs = []
  kFold = KFold(n_splits=k)
  print('K-Fold Cross Validation for k =', k)
  # Perform cross-validation
  for trainSet, testSet in kFold.split(trainData):
    # Train
    predictionAlgorithm.fit(trainSet)
    # Predict
    predictions = predictionAlgorithm.test(testSet)
    # Evaluate and print RSME for each fold
    rmse = accuracy.rmse(predictions, verbose=True)
    avgs.append(rmse)

  return np.average(avgs)