import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from surprise import accuracy
from preprocess.fileutil import readFile

def main():
    trainData = readFile('./src/data/train.data', separator=' ', columns=['userID', 'movieID', 'rating'], types={'userID': np.int32, 'movieID': np.int32, 'rating': np.float32})

    X = trainData.ix[:, 0:2]
    y = np.ravel(trainData.rating)

    #split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=(2,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse'])
    model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)

    y_pred = model.predict(X_test)
    score = model.evaluate(X_test, y_test, verbose=1)
    
    print(score)
    print(confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    main()