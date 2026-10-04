# Ethereum Anomaly Detection Web

A web-based Ethereum transaction anomaly detection system developed using Django and machine learning.

This application provides an interactive web interface for detecting anomaly-indicating Ethereum transactions using Random Forest and XGBoost models.

## Overview

This project is the web implementation of an Ethereum transaction anomaly detection research project.

The system allows users to:

- View an overview of Ethereum transaction data
- Perform single transaction prediction
- Perform batch prediction using CSV files
- View machine learning model information
- Review input features and evaluation metrics

The target variable `isError` is treated as an indicator of transaction execution error/anomaly and is **not interpreted as a fraud label**.

## Features

### 1. Dashboard

The dashboard provides an overview of the transaction dataset, including:

- Total transactions
- Number of anomaly-indicating transactions
- Number of normal transactions
- Anomaly rate

### 2. Single Transaction Prediction

Users can enter transaction information manually:

- From Address
- To Address
- Transaction Value
- Block Height
- Hour
- Machine learning model

The system then predicts whether the transaction is normal or anomaly-indicating.

### 3. Batch Analysis

Users can upload a CSV file containing multiple Ethereum transactions.

The system validates the uploaded file and processes the transactions using the selected machine learning model.

Required CSV columns:

```text
From
To
Value
BlockHeight
Hour
```
