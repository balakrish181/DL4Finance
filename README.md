# Bitcoin Price Prediction Using Deep Learning Techniques

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

This repository contains a comprehensive analysis and implementation of various deep learning techniques for Bitcoin price prediction. The project explores different approaches to time series forecasting in financial markets, with a focus on comparing traditional methods with advanced deep learning architectures.

## Key Features

- **Time Series Analysis**: Comprehensive characterization of Bitcoin price time series data, including stationarity tests, autocorrelation analysis, and statistical properties.
- **Multiple Transformation Techniques**: Implementation of various data transformation methods including:
  - Log returns for stationarity
  - Fractional differencing to preserve long-term memory while achieving stationarity
- **Advanced Deep Learning Models**:
  - Multilayer Perceptrons (MLPs) for raw and transformed time series
  - Convolutional Neural Networks (CNNs) with Gramian Angular Field (GAF) image representations
- **Hyperparameter Optimization**: Automated tuning of model hyperparameters using Keras Tuner
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations to compare different approaches

## Repository Structure

```
├── BTC_Price_Prediction_with_Deep_Learning.ipynb  # Main analysis notebook
├── LICENSE                                        # License file
└── README.md                                      # Project documentation
```

## Methodology

The project follows a structured approach to financial time series prediction:

1. **Data Collection and Preprocessing**: Bitcoin historical price data is collected, cleaned, and prepared for analysis.

2. **Time Series Characterization**: 
   - Statistical properties analysis
   - Stationarity testing using Augmented Dickey-Fuller test
   - Autocorrelation and partial autocorrelation analysis

3. **Data Transformation**:
   - Log returns calculation
   - Fractional differencing implementation
   - Gramian Angular Field (GAF) representation for CNN models

4. **Model Development**:
   - MLP models for raw price levels
   - MLP models for stationary time series
   - MLP models for fractionally differenced data
   - CNN models with GAF representations

5. **Performance Evaluation**:
   - Training and testing metrics (MAE, RMSE)
   - Visual comparison of predictions vs. actual values
   - Model performance analysis across different data transformations

## Key Findings

- Non-stationarity in raw Bitcoin price data significantly impacts prediction performance
- Fractional differencing provides a better balance between stationarity and information preservation compared to simple differencing
- GAF representation with CNN models captures complex temporal patterns effectively
- Hyperparameter optimization substantially improves model performance

## Getting Started

### Prerequisites

- Python 3.9+
- Required packages: tensorflow, keras, numpy, pandas, matplotlib, seaborn, statsmodels, scikit-learn, pyts, keras-tuner

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DL4Finance.git
cd DL4Finance

# Install required packages
pip install -r requirements.txt
```

### Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook BTC_Price_Prediction_with_Deep_Learning.ipynb
```

## Results

The project demonstrates that deep learning models, particularly CNNs with GAF representations, can effectively capture patterns in Bitcoin price movements. The fractional differencing approach provides a good balance between maintaining long-term memory and achieving stationarity, resulting in improved prediction performance.

## Future Work

- Incorporate additional features such as trading volume, market sentiment, and macroeconomic indicators
- Explore recurrent neural networks (RNNs) and transformer-based architectures
- Implement ensemble methods combining multiple model predictions
- Extend the analysis to other cryptocurrencies and traditional financial assets

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data sourced from Yahoo Finance
- Implementation inspired by recent advances in deep learning for financial time series prediction