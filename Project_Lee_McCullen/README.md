# CS 484 - Toxic Comment Classification with Convolutional Neural Networks

# Install Requirements
Make sure you are using Python 3.6 to run this project.

`pip3 install -r ./requirements.txt`

# Training and Test Files
Ensure `train.csv` and `test.csv` files are placed in the `/src/data`  directory before running the program.

These files can be retrieved from here: 
[Kaggle Link](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

# Pre-trained Embeddings
Currently, the `cnn.py` file uses pre-trained embeddings to train the CNN model. If you don't have the
following files, then the program will output an error. Our program uses GloVe's pre-trained embeddings
of 2 billion tweets. Please make sure the file `glove.twitter.27B.<embeddingDimensions>d` is placed in
the `/data` directory.

To download GloVe's pre-trained embeddings, click [here](https://nlp.stanford.edu/projects/glove/)

# Command Line Arguments
| Argument         | Description                                                      |
|:----------------:| ---------------------------------------------------------------- |
| `-t` or `--train` | Trains the model using `train.csv` and pre-trained embeddings |
| `-p` or `--predict` | Makes predictions from `test.csv` |
| `-s` or `--serve` | Host the model on local server to make real-time predictions |
| `-bs` or `--batchSize` | Batch size for training |
| `-ep` or `--epoch` | Epochs for training |
| `-ed` or `--embeddim` | Embedding dimensions of pre-trained or randomly-initialized embeddings |

# Running the Program
In the `/src` directory, the collaborative filtering method is implemented in `main.py`.

`python3 main.py [-t | -p | -s] -bs <batchSize> -ep <epochs> -ed <embeddingDimensions>`

The parameter `-ed <embeddingDimensions>` should match the same dimensions as the
pre-trained embeddings file. For example, if GloVe's pre-trained embeddings file is named as 
`glove.twitter.27B.100d`, then the the embeddingDimensions should be `100`.

# Results
The predictions will be written to `predicitions.data` located in `/src/data`.

# `/data` Directory and Subdirectories
Do not delete the`/src/data` directory as it will contain all of the files needed to run the toxic classifier. Place your training and test data in the `/src/data` directory.


