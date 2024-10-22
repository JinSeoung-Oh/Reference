### From https://generativeai.pub/financial-analysis-multi-agent-with-open-source-llms-using-crewai-and-ollama-models-9f20076f8995

## tools/comptiter_analysis.py
import yfinance as yf
from crewai_tools import tool

@tool
def competitor_analysis(ticker: str, num_competitors: int = 3):
    """
    Perform competitor analysis for a given stock.
    
    Args:
        ticker (str): The stock ticker symbol.
        num_competitors (int): Number of top competitors to analyze.
    
    Returns:
        dict: Competitor analysis results.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    sector = info.get('sector')
    industry = info.get('industry')
    
    # Get competitors in the same industry
    industry_stocks = yf.Ticker(f"^{sector}").info.get('components', [])
    competitors = [comp for comp in industry_stocks if comp != ticker][:num_competitors]
    
    competitor_data = []
    for comp in competitors:
        comp_stock = yf.Ticker(comp)
        comp_info = comp_stock.info
        competitor_data.append({
            "ticker": comp,
            "name": comp_info.get('longName'),
            "market_cap": comp_info.get('marketCap'),
            "pe_ratio": comp_info.get('trailingPE'),
            "revenue_growth": comp_info.get('revenueGrowth'),
            "profit_margins": comp_info.get('profitMargins')
        })
    
    return {
        "main_stock": ticker,
        "industry": industry,
        "competitors": competitor_data
    }

## tools/risk_assessment_tool.py
import yfinance as yf
import numpy as np
from scipy import stats
from crewai_tools import tool

@tool
def risk_assessment(ticker: str, benchmark: str = "^GSPC", period: str = "5y"):
    """
    Perform risk assessment for a given stock.
    
    Args:
        ticker (str): The stock ticker symbol.
        benchmark (str): Benchmark index for comparison (default: S&P 500).
        period (str): Time period for analysis.
    
    Returns:
        dict: Risk assessment results.
    """
    stock = yf.Ticker(ticker)
    benchmark_index = yf.Ticker(benchmark)
    
    stock_data = stock.history(period=period)['Close']
    benchmark_data = benchmark_index.history(period=period)['Close']
    
    # Calculate returns
    stock_returns = stock_data.pct_change().dropna()
    benchmark_returns = benchmark_data.pct_change().dropna()
    
    # Calculate beta
    covariance = np.cov(stock_returns, benchmark_returns)[0][1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance
    
    # Calculate Sharpe ratio
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    excess_returns = stock_returns - risk_free_rate
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    # Calculate Value at Risk (VaR)
    var_95 = np.percentile(stock_returns, 5)
    
    # Calculate Maximum Drawdown
    cumulative_returns = (1 + stock_returns).cumprod()
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
    
    return {
        "ticker": ticker,
        "beta": beta,
        "sharpe_ratio": sharpe_ratio,
        "value_at_risk_95": var_95,
        "max_drawdown": max_drawdown,
        "volatility": stock_returns.std() * np.sqrt(252)
    }


