from setuptools import setup, find_packages

setup(
    name="portfolio_optimization",
    version="0.1.0",
    description="Portfolio optimization with ML-enhanced parameter forecasting",
    author="Shivam Tyagi",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "yfinance>=0.2.3",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "cvxopt>=1.3.0",
        "pmdarima>=2.0.0",
        "arch>=5.0.0",
        "jupyter>=1.0.0",
        "pytest>=7.0.0",
        "black>=22.0.0",
        "pylint>=2.12.0",
    ],
    python_requires=">=3.8",
) 