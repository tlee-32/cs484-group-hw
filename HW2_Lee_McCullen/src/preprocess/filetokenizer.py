import os
import smart_open
import pickle
from nltk.tokenize import word_tokenize

"""
    Retrieves the label from the row (1 or 0).
    return - tuple(str, str)
"""
def splitLabelledRow(row):
    label = row[0]
    row = row[1:] 
    return label, row

"""
    Reads and tokenizes each row in the test/training file.

    return - tokenized rows and/or its associated labels
"""
def tokenizeReviews(fileName, isTrainingFile=False):
    tokens = []
    labels = []
    # read training file
    for row in smart_open.smart_open(fileName, encoding="utf-8"):
        label, row = splitLabelledRow(row)
        tokenizedRow = word_tokenize(row)
        # label training documents
        if isTrainingFile: labels.append(float(label))
        tokenizedRow = [float(item) for item in tokenizedRow]
        tokens.append(tokenizedRow)
    
    return tokens, labels

"""
    Read rows from raw training/test file OR load a pickled file to
    deserialize the object. Pickled files assume that the training/test file
    has already been read and checkpointed. Raw training/test files will be
    tokenized and pickled.

    return - tokenized rows a generator
"""
def readRows(fileName, loadFile=False, isTrainingFile=False):
    tokenFile = renameFileExtension(fileName, 'data', 'tokens')
    labelFile = renameFileExtension(fileName, 'data', 'labels')
    tokens, labels = [], []
    if(loadFile):
        # deserialize objects if pickled file already exists
        with smart_open.smart_open(tokenFile, "rb") as f:
            tokens = pickle.load(f, encoding="utf-8")
        if(isTrainingFile):
            with smart_open.smart_open(labelFile, "rb") as f: 
                labels = pickle.load(f, encoding="utf-8")
    else:
        # serialize and pickle the objects to files with pickled extension
        tokens, labels = tokenizeReviews(fileName, isTrainingFile) 
        serializeObject(tokenFile, tokens)
        if(isTrainingFile): serializeObject(labelFile, labels)
        
    return tokens, labels


"""
    Serializes the object into a file.
"""
def serializeObject(fileName, obj):
    with smart_open.smart_open(fileName, "wb") as f:
        pickle.dump(obj, f)

"""
    Deserializes the pickle file into an object
"""
def deserializeObject(fileName):
    with smart_open.smart_open(fileName, "rb") as f:
        return pickle.load(f)

def fileExists(fileName):
    return os.path.exists(fileName)

def renameFileExtension(fileName, oldExt, newExt):
    fileExtensionIdx = fileName.rfind(oldExt)
    newFileName = fileName[:fileExtensionIdx] + newExt
    return newFileName
