#!/usr/bin/env python
'''
Code for the study "A comparative study of classical machine learning algorithms for 
multi-class topic classification of English reviews"
Authors: Twan Huiskens, Koen Snelten and Tian Niezing

Train and evaluate different classifiers on a text classification task.

Usage:

python lfd_assignment1.py <command> [<args>]
Commands:

    split   Split the data into train, dev and test sets
    data    Collect information about the data
    train   Train and evaluate a model
    evaluate Evaluate a saved model on the test set

Examples:

Split the data into train, dev and test sents
`python lfd_assignment1.py split \
    reviews.txt train.txt dev.txt test.txt --seed 42`

Collect information about the data
`python lfd_assignment1.py data reviews.txt`

Train and evaluate a Naive Bayes model with tf-idf features and bigrams
`python lfd_assignment1.py train \
    nb train.txt dev.txt --tfidf --ngram 2 --save-model nb_model.pkl`

Perform grid search to find the best parameters for a Random Forest model
`python lfd_assignment1.py train \
    rf train.txt dev.txt --grid-search --save-model rf_model.pkl`

Evaluate a saved model on the test set
`python lfd_assignment1.py evaluate rf_model.pkl test.txt`
'''
import argparse
import pickle
from typing import TypeVar

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def create_arg_parser():
    
    parser = argparse.ArgumentParser()

    # Arguments for input files
    parser.add_argument('--train_file', default='./data/raw/train.tsv', type=str)
    parser.add_argument('--dev_file', default='./data/raw/dev.tsv', type=str)
    parser.add_argument('--test_file', default='./data/raw/test.tsv', type=str)

    # Argumen to set the  random seed
    parser.add_argument('--seed', default=42, type=int)


    # Save model flag
    # Saves the model to the given path (using pickle)
    parser.add_argument(
        '--save-model',
        help='Save the feature weights to a file',
        type=str,
        default=None,
    )

    # Parse and return the commandline args
    args = parser.parse_args()

    return args


def read_corpus(file):
    '''Reads the given corpus file and returns the documents and labels'''

    # Variables to store the documents and labels
    tweets = []
    labels = []

    # Open the file
    with open(file, encoding='utf-8') as in_file:
        # Read the file line by line
        for line in in_file:
            tweet, label = line.strip().split("\t")
            tweets.append(tweet)
            labels.append(label)
        
    return tweets, labels
 

def main() -> int:
    """
    Main function to run the program.

    The main function creates the argument parser and checks which commands
    were given. It then calls the appropriate function to handle the command.

    Returns 0 on success, 1 on failure.
    """

    # Create the arg parser
    args = create_arg_parser()

    # Define parameter grids for each model
    param_grid = {
        'cls__C': [0.1, 1, 10, 100],
        'cls__kernel': ['linear', 'rbf'],
        'cls__gamma': ['scale', 'auto'],
    }

    # Read the corpus and get the train and test data
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    X_test, Y_test = read_corpus(args.test_file)

    # Create the vectorizer and classifier based on the command line
    # arugments given by the user.
    vectorizer = TfidfVectorizer()
    classifier = SVC()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_dev_vec = vectorizer.transform(X_dev)

    # Create the pipeline that first vectorizes the data and then
    # applies the classifier.
    classifier = Pipeline(
        [
            ('vec', vectorizer),
            ('cls', classifier),
        ],
    )

    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, Y_train)

    # Show the results and best parameters
    print(f'Best parameters: {grid_search.best_params_}')
    classifier = grid_search.best_estimator_
    Y_pred = classifier.predict(X_dev)
    accuracy = accuracy_score(Y_dev, Y_pred)
    print("Dev scores")
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification report:')
    print(classification_report(Y_dev, Y_pred))

    print('Confusion Matrix:')
    labels = ['NOT', 'OFF']
    print(confusion_matrix(Y_dev, Y_pred, labels=labels))

    Y_pred_test = classifier.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred_test)
    print("Test scores")
    print(f'Accuracy: {test_accuracy:.4f}')
    print('Classification report:')
    print(classification_report(Y_test, Y_pred_test))

    print('Confusion Matrix:')
    print(confusion_matrix(Y_test, Y_pred_test, labels=labels))

    # Save the model using pickle if a path is given
    if args.save_model is not None:
        with open(args.save_model, 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)

if __name__ == '__main__':
    main()