# ANN-Based Binary Azeotropic VLE Prediction

An Artificial Neural Network (ANN) model for predicting vapor composition in the ethanol–water binary azeotropic Vapor-Liquid Equilibrium (VLE) system. The project demonstrates how machine learning can model complex non-ideal thermodynamic behavior that conventional thermodynamic models such as Raoult's Law cannot accurately represent.

---

## Overview

This project develops a deep learning model to predict the vapor composition of an ethanol–water azeotropic system using liquid composition, temperature, and pressure as input features. The model accurately captures nonlinear VLE behavior and identifies the azeotropic point with high precision.

---

## Features

- Predicts vapor composition using an Artificial Neural Network
- Models non-ideal ethanol–water VLE behavior
- Detects the azeotropic composition with high accuracy
- Includes model training, evaluation, and visualization
- Demonstrates machine learning for thermodynamic system modeling

---

## Dataset

A synthetic dataset consisting of **750 samples** was generated for the ethanol–water VLE system.

**Input Features**

- Liquid composition (x₁)
- Temperature (T)
- Pressure (P)

**Output**

- Vapor composition (y₁)

---

## Model Architecture

- Input Layer: 3 neurons
- Hidden Layer 1: 64 neurons
- Hidden Layer 2: 32 neurons
- Hidden Layer 3: 16 neurons
- Output Layer: 1 neuron

### Training Configuration

- Optimizer: Adam
- Hidden Layer Activation: ReLU
- Output Activation: Sigmoid
- L2 Regularization
- Batch Size: 32
- Train-Test Split: 80:20
- Validation Split: 20%
- Early Stopping Enabled

---

## Model Performance

| Metric | Value |
|---------|--------|
| Mean Absolute Error (MAE) | **0.011644** |
| Root Mean Square Error (RMSE) | **0.014573** |

### Azeotrope Prediction

| Parameter | Predicted |
|-----------|-----------|
| Composition | **0.8942** |
| Temperature | **78.07 °C** |

The model successfully detected the azeotropic composition with approximately **0.02% error**, demonstrating strong predictive capability.

---

## Results

The developed ANN model successfully:

- Learned nonlinear vapor-liquid equilibrium relationships
- Predicted vapor composition with low error
- Accurately detected the azeotropic point
- Converged without overfitting
- Outperformed traditional ideal-solution assumptions for the ethanol–water system

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## Visualizations

The project includes:

- Training loss curve
- Parity plot (Predicted vs Experimental)
- Prediction error distribution
- T-x-y diagram
- Azeotrope prediction visualization
