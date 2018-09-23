import smart_open
from preprocess.filetokenizer import readRows

def main():
    trainingRows, labels = readRows("./data/train_drugs.data", loadFile=False, isTrainingFile=True)
    testRows, _ = readRows("./data/test_drugs.data", loadFile=False, isTrainingFile=False)
    
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
