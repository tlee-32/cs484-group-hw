# CS 484 - Predict Active/Inactive Molecules (HW2)

# Install Requirements
Make sure you are using Python 3.6 to run this project.

`sudo pip3 install -r /requirements.txt`

# Training and Test Files
Ensure `train.data` and `test.data` files are placed in the `/src/data`  directory before running the program.

# Running the Program
In the `/src` directory, the only file you need to run is `moleculeclassifier.py`.

`python3 moleculeclassifier.py`

# Results
The sentiment predictions will be written to `predicitions.data` located in `/src/data`.

# Saving and Loading Files `.pkl`
A checkpoint is made for each stage of the pipeline:
Pre-processing -> Feature extraction/representation -> Cross-validation -> kNN classification

Checkpoints mean that data is pickled (saved) as a .pkl or .model file. In `moleculeclassifier.py`, the condition
`loadFile = True` if the file has already been saved and the `.pkl` file just needs to be loaded. If `loadFile = False`, then the `.data` file will be read and saved as a `.pkl` file.

# `/data` Directory and Subdirectories
Do not delete the`/src/data` directory as it will contain all of the files needed to run the sentiment classifier. Place your training and test data in the `/src/data` directory.
