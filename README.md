# Automatic Kinship Recognition

This is the code for Achieving Better Kinship Recognition Through Better Baseline.

### Requirements:

1. mxnet
1. insightface
1. gluonfr
1. OpenCV
1. matplotlib
1. tqdm

### Usage

There are two scripts that can be run: `verification.py` and `train.py`.
The first provides nessesary functions to prepare (as described in the section III-A of the paper) the dataset and run a validation and plot a ROC curve. The second one provides necessary code to reproduce our training procedure (the results can vary slightly).

### Data

We provide already prepared data (re-detected and re-aligned) that can be found in `train-faces-det` and `val-faces-det` folders for training and validation images respectfully.
