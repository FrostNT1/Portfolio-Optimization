# Portfolio Optimization Project

This project demonstrates an end-to-end portfolio optimization process combining traditional Mean-Variance Optimization with Machine Learning-based parameter forecasting.

## Project Structure

```
├── data/               # Raw and processed data
├── notebooks/          # Jupyter notebooks for analysis
├── src/               # Source code
│   ├── data/          # Data collection and preprocessing
│   ├── optimization/  # Portfolio optimization implementations
│   ├── ml/           # Machine learning models
│   └── utils/        # Utility functions
├── tests/             # Unit tests
├── config/            # Configuration files
└── results/           # Output and visualization results
```

## Features

1. Traditional Mean-Variance Optimization
   - Historical data analysis
   - Efficient frontier computation
   - Multiple constraint implementations
   - Rolling window backtesting

2. ML-Enhanced Portfolio Optimization
   - ARIMA/SARIMA return forecasting
   - GARCH volatility modeling
   - Dynamic covariance estimation
   - Comparative performance analysis

## Setup

1. Create a virtual environment:
```bash
conda create -n portfolio_optimization python=3.10
conda activate portfolio_optimization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Collection:
```bash
python src/data/collect_data.py
```

2. Run Optimization:
```bash
python src/optimization/optimize.py
```

3. Run Backtests:
```bash
python src/optimization/backtest.py
```

## Configuration

- Adjust parameters in `config/config.yaml`
- Modify asset universe in `config/universe.yaml`
- Configure constraints in `config/constraints.yaml`

## Testing

Run tests with:
```bash
pytest tests/
```

## License

MIT License 