# COMP-472-Project

The goal of this project is to create a Deep Learning Convolutional Neural Network (CNN) using PyTorch to analyze images of students in classroom or online meeting environments. The CNN classifies images into four categories based on facial expressions: Neutral, Engaged, Angry, and Happy.

## Content of the submission

<ins>src/data_cleaning.py</ins> : Python script for data cleaning of  dataset

<ins>src/visualization.py</ins>: Python script to create the data visualization

<ins>dataset_info.md</ins>: Document detailing the sources and licensing of  dataset (https://www.kaggle.com/datasets/msambare/fer2013)

<ins>dataset folder</ins>: Folder containing the dataset images

<ins>dataset-cleaned folder</ins>: Folder created after running data_cleaning.py that contains the cleaned dataset images

## How To Run Python Scripts

Setting Up the Virtual Environment

    1.	Navigate to the root directory of the project.
    2.	Create a virtual environment:
        • On Windows: py -m venv .venv
        • On macOS/Unix: python3 -m venv .venv
    3.	Activate the virtual environment:
        • On Windows: .venv\Scripts\activate
        • On macOS/Unix: source .venv/bin/activate
    4.	Install the required dependencies: pip install -r requirements.txt

### Steps for data cleaning + labeling

1. From the root folder, go to the src folder: `cd src`.
2. Run the data cleaning script `python data_cleaning.py` .
3. This will generate a folder named dataset-cleaned in the root directory containing the cleaned images.

### Data visualization

1. From the root folder, go to src folder
2. Run `python visualization.py`

### Training Models (Main Model and Variants)

1. From the root folder, go to src `cd src`.
2. Run `python split_dataset.py` to split dataset for training
   - Enter the relative path of the dataset you wish to use for training when prompted.
3. Train the models:
   - Main Model: enter `python trainAI_main.py`
   - Variant 1: enter `python variant1.py`
   - Variant 2: enter `python variant2.py`

### Steps to evaluate models

1. Ensure the models are trained using the steps above.
2. From the root folder, go to src folder `cd src`.
3. Run `python evaluation_models.py`

### Testing the Model on Dataset or Single Image

1. From the root folder, go to src `cd src`.
2. Run `python evaluation.py` to load the model.
3. Type "Dataset" to test the entire dataset or "single" to predict a single image.
4. If you typed "single", type the category of the image (angry, engaged, happy, or neutral).
5. If you typed "single", then type the full filepath of the image you wish to predict
   c. Variant 2: enter `python variant2.py`

### Evaluating Model Bias (Age and Gender)

1. From the root folder, go to src `cd src`.
2. You first need to split the training data.
   - a. Enter `python split_dataset.py` to split dataset.
   - b. Then you need to enter the relative path of the dataset (either one of the following):
     - Level 1: enter `../dataset-bias_level1/`
     - Level 2: enter `../dataset-bias_level2/`
     - Level 3: enter `../dataset-bias_level3/`
3. Create and train the bias models (either one of the following):
   - a. Level 1: enter `python trainAI_bias1.py`
   - b. Level 2: enter `python trainAI_bias2.py`
   - c. Level 3: enter `python trainAI_bias3.py`
4. Now, enter `python evaluation_bias.py` to evaluate the model based on the biases.
   - a. Enter the model name you would like to evaluate
     - Level 1: enter `model_bias1.pth`
     - Level 2: enter `model_bias2.pth`
     - Level 3: enter `model_bias3.pth`

### Steps to training with k-fold cross-validation

1. From the root folder, go to src fodler "cd src"
2. Run `python kfolding.py` to obtain and save the folds.
3. Train with k-fold cross-validation by running `python kfold_train.py`.

### Steps to evaluate the k-fold models.

1. From the root folder, go to src by typing `cd src`.
2. Enter `python evaluation_kfold.py` to obtain the performance metrics and confusion matrix.
