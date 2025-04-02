# AI Options Trading Analyst

An intelligent agent-based system for analyzing stock options and providing trading recommendations.

## Features

- Real-time stock data analysis using yfinance
- Technical analysis with multiple indicators (RSI, MACD, Bollinger Bands)
- Options chain analysis with volume and IV consideration
- News sentiment analysis
- Risk assessment and scoring
- Web interface using Gradio

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MouniReddy544/AgenticAI.git
cd AgenticAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python stock_analyzer.py
```

## Usage

1. Open the web interface (default: http://localhost:7860)
2. Enter a stock ticker (e.g., AAPL, SPY, NVDA)
3. Get comprehensive analysis including:
   - Technical indicators
   - Options opportunities
   - Market sentiment
   - Risk assessment

## Structure

- `stock_analyzer.py`: Main application file
- `requirements.txt`: Project dependencies
- `.gitignore`: Git ignore rules
- `.env`: Environment variables (not tracked in git)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 