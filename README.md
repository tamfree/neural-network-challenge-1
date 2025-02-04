# neural-network-challenge-1

## Description

Create a model to predict student loan repayment.

## Execution

### Dependencies

Python >= version 3.9

[Pandas](https://pandas.pydata.org/)

[Scikit-learn](https://scikit-learn.org/)

* sklearn.preprocessing.StandardScaler
* sklearn.model_selection.train_test_split
* sklearn.metrics.accuracy_score
* sklearn.linear_model.LogisticRegression
* sklearn.ensemble.RandomForestClassifier

[TensorFlow Keras](https://www.tensorflow.org/guide/keras)
* tensorflow.keras.models.Sequential
* tensorflow.keras.layers.Dense

### Data source(s)

Dataset Source: [Simulated business data representing loan repayment histories](https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv)

### Command

To run locally (refer to TensorFlow documentation to determine if this appropriate for your machine).

        conda install tensorflow
        or
        pip install tensorflow

Open student_loans_with_deep_learning.ipynb in your preferred Jupyter Notebook editor.

## Process

The below process was roughly followed in spam_detector.ipynb

1. Ingest data using Pandas
1. Review data.
1. Split data into training and test datasets 
1. Normalize the features by adjusting the scale of the numeric features using sklearn.preprocessing.StandardScaler
1. Instantiate and train models
1. Create test predicitions
1. Calculate accuracy scores
1. Adjust the model to improve scores. Optionally, return to step 5.

Predictions were limited to test data for this exercise.

<details>
    <summary> Associated Lesson</summary>

## Lesson 18 - Neural Networks 1

Concepts covered:

* Splitting data set into training and test data sets using sklearn.model_selection.train_test_split
* Using *Standard Scaling* to normalize data.
* Using *Accuracy Scores* to determing if a model is likely to yield an accurate prediction for the specified dataset
* Using Tensorflow Keras
* Using the *tensorflow.keras.models.Sequential*
* Persisting and loading trained keras models.
* Recommentation systems and associated filtering

</details>