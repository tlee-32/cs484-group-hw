from sklearn.model_selection import cross_val_score

def logRegCrossValidation(logRegClassifier, tokens, labels, folds, scoringMethod):
    return cross_val_score(logRegClassifier, tokens, labels, cv=folds, scoring=scoringMethod)
