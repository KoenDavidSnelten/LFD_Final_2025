#!/usr/bin/env python
'''
Finetunes a Transformer model for text classification with early stopping.
'''
import argparse

import numpy as np
from datasets import Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers import Trainer
from transformers import TrainingArguments


def create_arg_parser():
    """Creates the argument parser for command line arguments."""

    p = argparse.ArgumentParser(
        description='Fine-tune a Transformer with Trainer',
    )

    # Data / model arguments
    p.add_argument('--model_name', default='bert-base-uncased')
    p.add_argument('--train_file', default='train.txt')
    p.add_argument('--dev_file', default='dev.txt')
    p.add_argument('--test_file', default=None)
    p.add_argument('--confusion_matrix', default=None)
    p.add_argument('--max_length', type=int, default=100)

    # Core TrainingArguments
    p.add_argument('--output_dir', default='./out')
    p.add_argument('--num_train_epochs', type=int, default=1)
    p.add_argument('--per_device_train_batch_size', type=int, default=8)
    p.add_argument('--per_device_eval_batch_size', type=int, default=8)
    p.add_argument('--learning_rate', type=float, default=5e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument(
        '--eval_strategy',
        choices=['no', 'steps', 'epoch'], default='epoch',
    )
    p.add_argument('--logging_steps', type=int, default=50)
    p.add_argument('--warmup_ratio', type=float, default=0.0)
    p.add_argument('--warmup_steps', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)

    p.add_argument(
        '--early_stopping_patience', type=int,
        default=3, help='Patience for early stopping.',
    )
    p.add_argument(
        '--metric_for_best_model', type=str,
        default='eval_f1_macro', help='Metric to monitor for best model.',
    )
    p.add_argument(
        '--load_best_model_at_end', action='store_true',
        help='Load the best model at the end of training.',
    )

    return p.parse_args()


def read_corpus(corpus_file):
    '''Read in review data set and returns docs and labels'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip()
            documents.append(' '.join(tokens.split()[3:]).strip())
            labels.append(tokens.split()[0])
    return documents, labels


def save_confusion_matrix(cm, le, filename):
    """Plot and save confusion matrix with the labels using matplotlib."""

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    # Get the label names to show in the matrix
    labels = le.classes_
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', xticks_rotation='vertical', values_format='d')
    plt.tight_layout()
    # Save the figure
    plt.savefig(filename)
    plt.close()


def compute_metrics(eval_pred):
    '''Metrics we compute for the dev set'''
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'f1_micro': f1_micro, 'f1_macro': f1_macro}


def print_results(metrics):
    '''Print final results for the dev set here'''
    # The metric keys will have an "eval_" prefix
    acc = round(metrics['eval_accuracy'] * 100, 1)
    f1_micro = round(metrics['eval_f1_micro'] * 100, 1)
    f1_macro = round(metrics['eval_f1_macro'] * 100, 1)
    print(metrics)
    print('\n\nFinal metrics:\n')
    print(f'Accuracy: {acc}')
    print(f'Micro F1: {f1_micro}')
    print(f'Macro F1: {f1_macro}')


def prepare_data(args, tokenizer):
    '''Tokenize and build datasets + label encoder'''
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    le = LabelEncoder()
    y_train_ids = le.fit_transform(Y_train).astype(np.int64)
    y_dev_ids = le.transform(Y_dev).astype(np.int64)

    tok_train = tokenizer(
        X_train, padding=True,
        truncation=True, max_length=args.max_length,
    )
    tok_dev = tokenizer(
        X_dev, padding=True, truncation=True,
        max_length=args.max_length,
    )

    train_ds = Dataset.from_dict(
        {**tok_train, 'labels': y_train_ids},
    ).with_format('torch')
    dev_ds = Dataset.from_dict(
        {**tok_dev, 'labels': y_dev_ids},
    ).with_format('torch')

    return train_ds, dev_ds, le


def main():
    '''Main function'''
    args = create_arg_parser()

    # Create the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Load and prepare the datasets
    train_ds, dev_ds, le = prepare_data(args, tokenizer)
    num_labels = len(le.classes_)
    # Create the model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels,
    )

    # Define training arguments based on given command line args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.eval_strategy,

        save_strategy=args.eval_strategy,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio if args.warmup_steps == 0 else 0.0,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        report_to=[],

        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        save_total_limit=2,
    )

    # Implement early stopping
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    # Start training and evaluate on the dev set
    trainer.train()
    metrics = trainer.evaluate()
    print_results(metrics)

    # Evaluate on the test set if provided
    if args.test_file:
        # Load and process the test dataset
        X_test, Y_test = read_corpus(args.test_file)
        y_test_ids = le.transform(Y_test).astype(np.int64)
        tok_test = tokenizer(
            X_test, padding=True, truncation=True,
            max_length=args.max_length,
        )
        test_ds = Dataset.from_dict(
            {**tok_test, 'labels': y_test_ids},
        ).with_format('torch')

        # Get predictions
        test_metrics = trainer.evaluate(eval_dataset=test_ds)
        print('\nTest set results:')
        print_results(test_metrics)

        # If specified, create and save the confusion matrix image
        if args.confusion_matrix is not None:
            # Print confusion matrix for the test file
            preds_output = trainer.predict(test_ds)
            preds = np.argmax(preds_output.predictions, axis=-1)
            cm = confusion_matrix(y_test_ids, preds)
            print('Confusion Matrix:')
            print(cm)
            save_confusion_matrix(cm, le, args.confusion_matrix)
            print(f'Saving confusion matrix to {args.confusion_matrix}')


if __name__ == '__main__':
    main()