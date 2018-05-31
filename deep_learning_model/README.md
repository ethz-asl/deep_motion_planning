# deep_learning_model

This part of the code is responsible for the imitation learning of the navigation model based on expert demonstrations. 
`model.py` in src->model contains a simple fully connected and a more complex CNN model that can be used to represent the navigation policy. 

The model training was successfully tested with Tensorflow 1.6.0.

## Usage
### Data Generation
Run:
```
make data
```
to take the single .csv files and generate a single HDF5 container with the entire data. Please
refer to the Makefile to change the files and folders that are used for the data preparation.

### Train a Model
```
make train
```
This will start the training of your model. Please refer to the Makefile to change the files and folders
that are used during training. Furthermore, we set some metaparameters (initial learning rate,
maximum number of steps and the batch size) in this file and you can choose between the different models (simple fully connected or convolutional model) by setting the option `--use_conv_model` or leaving it blank. This will 

```
make sync-gpu
```
This will sync this directory with the GPU of choice. The GPU credentials (username and IP address) can either be entered in the Makefile or via the command line.  

### Tensorboard

```
make tensorboard
```
This will start Googles Tensorboard with the ./models/default folder as log directory.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate and format data
        │   └── make_dataset.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
