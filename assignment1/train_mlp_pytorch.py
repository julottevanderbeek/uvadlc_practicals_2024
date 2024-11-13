################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    predicted_labels = predictions.argmax(dim=1)
    correct = (predicted_labels == targets).float()
    accuracy = correct.mean()
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    model.eval()
    accuracy_total = 0
    samples_total = 0 
    
    with torch.no_grad():
      for batch in data_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        inputs = inputs.view(inputs.size(0), -1)
        outputs = model(inputs)
        accuracy_batch = accuracy(outputs, labels)
        accuracy_total += accuracy_batch * inputs.size(0)
        samples_total += inputs.size(0)
      avg_accuracy = accuracy_total / samples_total
      
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation.
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    train_loader = cifar10_loader['train']
    val_loader = cifar10_loader['validation']
    test_loader = cifar10_loader['test']
    
    # TODO: Initialize model and loss module
    n_inputs = 3*32*32
    model = MLP(n_inputs, hidden_dims, n_classes=10, use_batch_norm=use_batch_norm).to(device)
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # TODO: Training loop including validation
    val_accuracies = []
    train_losses = []
    best_val_accuracy = 0.0
    best_model = None
    
    for epoch in range(epochs):
      epoch_losses = []
      model.train()
      
      # Training
      for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.view(inputs.size(0), -1)
        
        # Forward pass
        predictions = model(inputs)
        loss = loss_module(predictions, targets)
        epoch_losses.append(loss.item())
        
        # TODO: Do optimization with the simple SGD optimizer
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
      train_losses.append(np.mean(epoch_losses))
      val_accuracy = evaluate_model(model, val_loader)
      val_accuracies.append(val_accuracy)
      
      # Save best model
      if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = deepcopy(model)
        
      print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_losses[-1]:.4f}, Validation Accuracy: {val_accuracy:.4f}")
      
    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, test_loader)
    print(f"Test Accuracy of best model: {test_accuracy:.4f}")
    
    # TODO: Add any information you might want to save for plotting
    logging_dict = {
      'train_losses': train_losses,
      'val_accuracies': val_accuracies,
      'best_val_accuracy': best_val_accuracy,
      'test_accuracy': test_accuracy
    }
    
    # Convert tensors to CPU if training on Snellius for batchnorm 
    # train_losses_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    # val_accuracies_cpu = [acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in val_accuracies]

    # Plot loss and accuracy
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    #plt.plot(train_losses_cpu, label='Training Loss')
    plt.plot(logging_dict['train_losses'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(logging_dict['val_accuracies'], label='Validation Accuracy', color='orange')
    #plt.plot(val_accuracies_cpu, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves_pytorch.png")
    plt.show()
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
