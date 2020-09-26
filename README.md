# Classifiers - Machine Learning

Python implementation from scratch of machine learning algorithms used for classification.

The training set with ***N*** elements is defined as ***D={(X1, y1), . . ., (XN, yN)}***, where ***X*** is a vector and ***y={0, 1}*** is one-hot encoded.

## Contents
**autoencoder:** Autoencoder with sigmoid activation function in 2nd and 4th layers.

**neural convolutional network:** 
Flexible architecture of Convolutional Neural Network, with sigmoid and relu activation functions. Setup the number of layers:
- Convolution layer.
- Pooling layer.
- Full connected layer.

**ensemble:** Implementation of three ensemble methods of neural networks:
1. Ensemble learing via negative correlation.
1. Ensemble Learning Using Decorrelated Neural Networks.
1. Creating Diversity In Ensembles Using Artiflcial Data (DECORATE).

**mixture of experts:** Two setups of mixture of experts for time series:
1. Linear models for experts and gating with softmax output.
1. Linear models for experts and gating with normalized gaussian output.

**neural networks:** Single Layer Perceptron (SLP) and Multi Layer Perceptron (MLP).

**optimization:** first and second order methods used in machine learning backpropagation. Methods available:

1. Gradient Descent.
1. Bisection.
1. Newton.
1. Modified Newton.
1. Levenberg-Marquardt.
1. Quasi-Newton Methods.
1. One Step Secant.
1. Conjugate Gradient Methods.

**regression:** implementation of three setups of regression for classification:
1. Linear regression.
1. Linear regression with regularization.
1. Logistic regression.

**svm:** implementation of three models of Support Vector Machines for binary and multi-class classification.
1. Traditional Support Vector Machines (SVM).
1. Least Squares Support Vector Machines (LSSVM).
1. Twin Support Vector Machines (TWSVM).

kernel types:
1. Linear
1. Polynomial
1. Radial Base Function (RBF)
1. Exponential Radial Base Function (ERBF)
1. Hyperbolic tangent (tanh)
1. Linear splines
