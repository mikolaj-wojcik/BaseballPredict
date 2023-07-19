
import tensorflow as tf

import pandas as pd
import numpy as np

import  joblib

# Note: this notebook requires torch >= 1.10.0
import keras

def loadModel():
    model = tf.keras.models.load_model('C:/Users/Mikolaj/Downloads/MLBData/model.keras')
    return model

def loadCsvAndNormalize(path):
    dat = pd.read_csv(path)
    min_max_scaler = joblib.load('C:/Users/Mikolaj/Downloads/MLBData/MinMaxScaler.save')
    dat = min_max_scaler.transform(dat)
    return dat

def predictOutcome(model, data):
    outcome = model(data, training = False)
    out_df = pd.DataFrame(outcome.numpy())
    results = out_df.idxmax(axis =1)
    resultsLetters = []
    for i in range(len(results.index)):
        column = results[i]

        if(column == 0):
            resultsLetters.append('B')
        if (column == 1):
            resultsLetters.append('S')
        if (column == 2):
            resultsLetters.append('X')
    return resultsLetters

def printOutcomes(outcomes):
    print ('Predicted:')
    for single in outcomes:
        print(single)

def printOutcomesWithReal(outcomes, outReal):
    letters =outReal.idxmax(axis =1).tolist()
    print(letters)
    print('Predicted Real:')
    for (pred, real) in zip(outcomes, letters):
        print(pred, real)


def userInterface(model):
    path = input('Path to csv data of baseball pitch\n')
    answer = ''
    outPath = ''
    while(answer != 'y' and answer!= 'n'):
        answer = input('\nDo you want to load original outcome? (y/n)\n')
    if(answer == 'y'):
        outPath = input('\nPath to results of pitches: \n')
        outReal = pd.read_csv(outPath)

    data = loadCsvAndNormalize(path)
    outcomes = predictOutcome(model, data)
    if (answer == 'y'):
        printOutcomesWithReal(outcomes, outReal)
    else:
        printOutcomes(outcomes)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = loadModel()
    userInterface(model)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
