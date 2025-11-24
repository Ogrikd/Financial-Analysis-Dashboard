# Financial Analysis Dashboard

A comprehensive Python-based financial analysis tool for analyzing stock market data with technical indicators, pattern recognition, and risk-return metrics.

## Features

- Data Loading & Processing: Fetch and process financial data for multiple stocks
- Technical Indicators: Calculate key indicators including:
  - Moving Averages (SMA, EMA)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  
- Pattern Recognition: Identify common chart patterns and trading signals
- Risk-Return Analysis: Evaluate portfolio performance metrics
- Interactive Dashboard: Visualize data and analysis results

## Project Structure

```
src/
├── analysis.py              # Core analysis functions
├── data_loader.py           # Data fetching and loading utilities
├── indicator.py             # Technical indicator calculations
├── pattern_recognition.py   # Chart pattern detection
├── risk_return_analysis.py  # Risk and return metrics
├── dashboard.py             # Dashboard visualization
└── Stocks Requested.csv     # Stock symbols configuration
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ogrikd/Financial-Analysis-Dashboard.git
cd Financial-Analysis-Dashboard
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
# Run the dashboard
python dashboard.py
```

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- yfinance (or your data source library)
- Additional dependencies as needed

## Future Enhancements
- Add more technical indicators
- Implement backtesting functionality
- Expand pattern recognition algorithms
- Add portfolio optimization features
- Improve dashboard interactivity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

  Ogriki Moses - Ogrikim@gmail.com

Project Link: [https://github.com/Ogrikd/Financial-Analysis-Dashboard](https://github.com/Ogrikd/Financial-Analysis-Dashboard)
