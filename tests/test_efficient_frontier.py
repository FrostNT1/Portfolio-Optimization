import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_optimization.optimization import PortfolioOptimizer

@pytest.fixture
def sample_returns():
    """Generate sample returns data with realistic properties."""
    np.random.seed(42)
    n_days = 252
    n_assets = 5
    
    # Create correlation matrix with realistic correlations
    corr = np.array([[1.0, 0.5, 0.3, 0.2, 0.1],
                     [0.5, 1.0, 0.4, 0.3, 0.2],
                     [0.3, 0.4, 1.0, 0.5, 0.3],
                     [0.2, 0.3, 0.5, 1.0, 0.4],
                     [0.1, 0.2, 0.3, 0.4, 1.0]])
    
    # Generate means and standard deviations (more realistic values)
    means = np.array([0.0004, 0.0003, 0.0002, 0.00025, 0.00035])  # Daily returns (~10%, 7.5%, 5%, 6.3%, 8.8% annually)
    stds = np.array([0.01, 0.008, 0.012, 0.009, 0.011])  # Daily volatilities (~16%, 13%, 19%, 14%, 17% annually)
    
    # Generate correlated returns
    L = np.linalg.cholesky(corr)
    uncorrelated = np.random.normal(0, 1, (n_days, n_assets))
    correlated = uncorrelated @ L.T
    
    # Scale returns to have desired means and standard deviations
    returns = pd.DataFrame(
        correlated * stds + means,
        columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    )
    
    # Verify the properties
    print("\nGenerated Returns Properties (Annualized):")
    print("Means:")
    print((returns.mean() * 252).to_string())
    print("\nVolatilities:")
    print((returns.std() * np.sqrt(252)).to_string())
    print("\nCorrelations:")
    print(returns.corr().round(3).to_string())
    
    return returns

@pytest.fixture
def optimizer():
    """Create optimizer instance with test configuration."""
    return PortfolioOptimizer('tests/test_config.yaml')

def test_sample_returns_properties(sample_returns):
    """Test that sample returns have expected statistical properties."""
    # Check basic properties
    assert isinstance(sample_returns, pd.DataFrame)
    assert len(sample_returns.columns) == 5
    assert len(sample_returns) == 252
    
    # Check that returns are in reasonable range
    assert np.all(np.abs(sample_returns.mean()) < 0.01)  # Daily means should be small
    assert np.all(sample_returns.std() < 0.1)  # Daily volatilities should be reasonable
    
    # Check correlation structure
    corr = sample_returns.corr()
    assert 0.4 < corr.iloc[0, 1] < 0.6  # Check correlation between first two assets
    assert 0.2 < corr.iloc[0, 2] < 0.4  # Check correlation between first and third assets

def test_minimum_volatility_portfolio(optimizer, sample_returns):
    """Test minimum volatility portfolio optimization."""
    result = optimizer.optimize_portfolio(sample_returns, objective='volatility')
    
    # Check result structure
    assert isinstance(result, dict)
    assert all(key in result for key in ['weights', 'return', 'volatility', 'sharpe_ratio'])
    
    # Check weights
    weights = np.array(list(result['weights'].values()))
    assert np.isclose(np.sum(weights), 1.0, atol=1e-3)
    assert all(w >= optimizer.constraints['min_position'] - 1e-3 for w in weights)
    assert all(w <= optimizer.constraints['max_position'] + 1e-3 for w in weights)
    
    # Check metrics
    assert result['volatility'] > 0
    assert np.isfinite(result['return'])
    assert np.isfinite(result['sharpe_ratio'])

def test_maximum_return_portfolio(optimizer, sample_returns):
    """Test maximum return portfolio optimization."""
    result = optimizer.optimize_portfolio(sample_returns, objective='return')
    
    # Check result structure
    assert isinstance(result, dict)
    assert all(key in result for key in ['weights', 'return', 'volatility', 'sharpe_ratio'])
    
    # Check weights
    weights = np.array(list(result['weights'].values()))
    assert np.isclose(np.sum(weights), 1.0, atol=1e-3)
    assert all(w >= optimizer.constraints['min_position'] - 1e-3 for w in weights)
    assert all(w <= optimizer.constraints['max_position'] + 1e-3 for w in weights)
    
    # Check metrics
    assert result['return'] > 0
    assert result['volatility'] > 0
    assert np.isfinite(result['sharpe_ratio'])

def test_maximum_sharpe_portfolio(optimizer, sample_returns):
    """Test maximum Sharpe ratio portfolio optimization."""
    result = optimizer.optimize_portfolio(sample_returns, objective='sharpe')
    
    # Check result structure
    assert isinstance(result, dict)
    assert all(key in result for key in ['weights', 'return', 'volatility', 'sharpe_ratio'])
    
    # Check weights
    weights = np.array(list(result['weights'].values()))
    assert np.isclose(np.sum(weights), 1.0, atol=1e-3)
    assert all(w >= optimizer.constraints['min_position'] - 1e-3 for w in weights)
    assert all(w <= optimizer.constraints['max_position'] + 1e-3 for w in weights)
    
    # Check metrics
    assert result['return'] > 0
    assert result['volatility'] > 0
    assert result['sharpe_ratio'] > 0

def test_efficient_frontier_generation(optimizer, sample_returns):
    """Test efficient frontier generation with Monte Carlo simulation."""
    # Generate efficient frontier with fewer points for testing
    ef = optimizer.generate_efficient_frontier(sample_returns, n_points=5, show_plots=False)
    
    # Check basic properties
    assert isinstance(ef, pd.DataFrame)
    assert not ef.empty
    assert all(col in ef.columns for col in ['return', 'volatility', 'sharpe_ratio', 'weights'])
    
    # Check that returns are ordered
    assert ef['return'].is_monotonic_increasing or ef['return'].is_monotonic_decreasing
    
    # Check that all values are finite
    assert np.all(np.isfinite(ef['return']))
    assert np.all(np.isfinite(ef['volatility']))
    assert np.all(np.isfinite(ef['sharpe_ratio']))
    
    # Check that weights for each portfolio sum to 1 and satisfy constraints
    for _, row in ef.iterrows():
        weights = np.array(list(row['weights'].values()))
        assert np.isclose(np.sum(weights), 1.0, atol=1e-3)
        assert all(w >= optimizer.constraints['min_position'] - 1e-3 for w in weights)
        assert all(w <= optimizer.constraints['max_position'] + 1e-3 for w in weights)
    
    # Test plotting functionality
    fig1 = optimizer.plot_efficient_frontier(ef, show_sharpe=True, show_assets=True, returns=sample_returns, show_plot=False)
    plt.close(fig1)
    
    # Test weight plotting functionality
    fig2 = optimizer.plot_portfolio_weights(ef, n_points=5, show_plot=False)
    plt.close(fig2)

def test_monte_carlo_simulation(optimizer, sample_returns):
    """Test Monte Carlo simulation in efficient frontier plotting."""
    # Generate efficient frontier
    ef = optimizer.generate_efficient_frontier(sample_returns, n_points=5, show_plots=False)
    
    # Plot with Monte Carlo simulation
    fig = optimizer.plot_efficient_frontier(ef, show_sharpe=True, show_assets=True, returns=sample_returns, show_plot=False)
    
    # Check that we have scatter plots (Monte Carlo points and assets)
    scatter_plots = []
    line_plots = []
    
    # Check all axes in the figure
    for ax in fig.get_axes():
        scatter_plots.extend([child for child in ax.get_children() if isinstance(child, plt.matplotlib.collections.PathCollection)])
        line_plots.extend([child for child in ax.get_children() if isinstance(child, plt.matplotlib.lines.Line2D)])
    
    # Should have at least Monte Carlo points and assets
    assert len(scatter_plots) >= 2
    # Should have efficient frontier line
    assert len(line_plots) >= 1
    
    plt.close(fig)

def test_efficient_frontier_uniqueness(optimizer, sample_returns):
    """Test the uniqueness analysis of efficient frontier points."""
    # Generate efficient frontier with potential duplicates
    ef = optimizer.generate_efficient_frontier(sample_returns, n_points=10)
    
    # Analyze uniqueness
    unique_ef = optimizer.analyze_ef_uniqueness(ef)
    
    # Check that unique points are properly filtered
    assert len(unique_ef) <= len(ef)
    assert not unique_ef.empty
    
    # Check that remaining points are unique when rounded
    rounded_points = unique_ef[['return', 'volatility']].round(4)
    assert not rounded_points.duplicated().any()
    
    # Check that point-to-point differences are calculated
    assert 'return_diff' in unique_ef.columns
    assert 'volatility_diff' in unique_ef.columns

def test_efficient_frontier_relaxed(optimizer, sample_returns):
    """Test efficient frontier generation with relaxed constraints."""
    # Generate efficient frontier with relaxed constraints
    ef_relaxed = optimizer.generate_efficient_frontier_relaxed(sample_returns, n_points=5)
    
    # Check basic properties
    assert isinstance(ef_relaxed, pd.DataFrame)
    assert not ef_relaxed.empty
    
    # Check that constraints are properly relaxed
    for _, row in ef_relaxed.iterrows():
        weights = np.array(list(row['weights'].values()))
        # Sum to 1 constraint should still hold
        assert np.isclose(np.sum(weights), 1.0, atol=1e-3)
        # But weights can be more extreme
        assert all(w >= 0.0 for w in weights)  # Only non-negativity constraint
        assert all(w <= min(0.5, 2.0 / len(sample_returns.columns)) + 1e-3 for w in weights)

def test_efficient_frontier_boundary_portfolios(optimizer, sample_returns):
    """Test that efficient frontier includes or approximates min vol and max return portfolios."""
    # Get boundary portfolios
    min_vol_port = optimizer.optimize_portfolio(sample_returns, objective='volatility')
    max_ret_port = optimizer.optimize_portfolio(sample_returns, objective='return')
    
    # Generate efficient frontier
    ef = optimizer.generate_efficient_frontier(sample_returns, n_points=10)
    
    # Check that frontier spans the range between min vol and max return
    assert np.isclose(ef['return'].min(), min_vol_port['return'], rtol=0.1)
    assert np.isclose(ef['return'].max(), max_ret_port['return'], rtol=0.1)
    
    # Check that min vol portfolio has lowest volatility
    assert np.isclose(ef['volatility'].min(), min_vol_port['volatility'], rtol=0.1) 