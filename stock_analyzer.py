import yfinance as yf
import pandas as pd
import talib
import numpy as np
from datetime import datetime, timedelta
import gradio as gr
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any
import logging

# Add these after the imports
ALPHA_VANTAGE_API_KEY = "your_key_here"
NEWS_API_KEY = "your_key_here"
SEEKING_ALPHA_API_KEY = "your_key_here"

class StockAgent:
    def __init__(self):
        self.memory = []
        self.confidence_threshold = 0.7
        self.data_sources = {
            "market_data": ["yfinance"],
            "technical_tools": ["talib"]
        }

    def gather_information(self, ticker: str) -> Dict[str, Any]:
        """Agent gathers information from multiple sources"""
        try:
            logging.info(f"Agent gathering information for {ticker}")
            data = {
                "market_data": self._get_market_data(ticker),
                "news_sentiment": self._analyze_news_sentiment(ticker),
                "options_chain": self._get_options_chain(ticker),
                "technical_signals": self._analyze_technical_signals(ticker)
            }
            self.memory.append({"timestamp": datetime.now(), "data": data})
            return data
        except Exception as e:
            logging.error(f"Error gathering information for {ticker}: {str(e)}")
            raise Exception(f"Failed to analyze {ticker}. Please check if the ticker is valid.")

    def _get_market_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive market data including options"""
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")['Close'][-1]
        
        # Get options chain for next few expirations
        options = stock.options[:4]  # Get next 4 expiration dates
        calls_data = []
        puts_data = []
        
        for expiration in options:
            opt = stock.option_chain(expiration)
            calls_data.append(opt.calls)
            puts_data.append(opt.puts)
            
        return {
            "current_price": current_price,
            "options_expirations": options,
            "calls_data": calls_data,
            "puts_data": puts_data
        }

    def _analyze_news_sentiment(self, ticker: str) -> Dict[str, float]:
        """Analyze news sentiment using yfinance news data"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            # Simple sentiment based on title keywords
            positive_words = ['up', 'rise', 'gain', 'positive', 'bull', 'growth']
            negative_words = ['down', 'fall', 'loss', 'negative', 'bear', 'decline']
            
            sentiment_score = 0.5  # neutral default
            if news:
                positive_count = 0
                negative_count = 0
                for item in news[:10]:  # analyze last 10 news items
                    title = item['title'].lower()
                    positive_count += sum(1 for word in positive_words if word in title)
                    negative_count += sum(1 for word in negative_words if word in title)
                
                if positive_count + negative_count > 0:
                    sentiment_score = positive_count / (positive_count + negative_count)
                
            return {
                "sentiment_score": sentiment_score,
                "confidence": 0.7 if news else 0.5
            }
        except Exception as e:
            logging.warning(f"News sentiment analysis failed: {str(e)}")
            return {"sentiment_score": 0.5, "confidence": 0.3}

    def _get_options_chain(self, ticker: str) -> Dict[str, Any]:
        """Analyze options chain for opportunities"""
        stock = yf.Ticker(ticker)
        
        # Get nearest expiration
        next_expiration = stock.options[0]
        chain = stock.option_chain(next_expiration)
        
        # Analyze implied volatility and volume
        calls = chain.calls
        puts = chain.puts
        
        return {
            "expiration": next_expiration,
            "high_volume_calls": calls[calls['volume'] > calls['volume'].quantile(0.8)],
            "high_volume_puts": puts[puts['volume'] > puts['volume'].quantile(0.8)],
            "iv_skew": self._calculate_iv_skew(calls, puts)
        }

    def _calculate_iv_skew(self, calls: pd.DataFrame, puts: pd.DataFrame) -> float:
        """Calculate implied volatility skew"""
        atm_call = calls.iloc[(calls['strike'] - calls['strike'].mean()).abs().argsort()[:1]]
        atm_put = puts.iloc[(puts['strike'] - puts['strike'].mean()).abs().argsort()[:1]]
        
        return float(atm_put['impliedVolatility'].iloc[0] - atm_call['impliedVolatility'].iloc[0])

    def _analyze_technical_signals(self, ticker: str) -> Dict[str, Any]:
        """Enhanced technical analysis with options focus"""
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")
        
        # Calculate technical indicators
        df['RSI'] = talib.RSI(df['Close'])
        df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'])
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
        
        return {
            "rsi": df['RSI'].iloc[-1],
            "macd_signal": "Buy" if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "Sell",
            "volatility": df['ATR'].iloc[-1],
            "bollinger_position": self._get_bollinger_position(df)
        }

    def _get_bollinger_position(self, df: pd.DataFrame) -> str:
        last_close = df['Close'].iloc[-1]
        if last_close > df['BB_Upper'].iloc[-1]:
            return "Overbought"
        elif last_close < df['BB_Lower'].iloc[-1]:
            return "Oversold"
        return "Neutral"

    def make_trading_decision(self, ticker: str) -> Dict[str, Any]:
        """Agent makes final trading decision based on all available data"""
        data = self.gather_information(ticker)
        
        # Analyze options opportunities
        options_analysis = self._analyze_options_opportunities(data)
        
        # Generate trading recommendation
        recommendation = self._generate_recommendation(data, options_analysis)
        
        return {
            "recommendation": recommendation,
            "confidence": self._calculate_confidence(data),
            "supporting_data": data,
            "timestamp": datetime.now()
        }

    def _analyze_options_opportunities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze specific options trading opportunities"""
        market_data = data["market_data"]
        technical = data["technical_signals"]
        technical['current_price'] = market_data['current_price']  # Add current price to technical data
        
        opportunities = []
        for i, expiration in enumerate(market_data["options_expirations"]):
            calls = market_data["calls_data"][i]
            puts = market_data["puts_data"][i]
            
            # Process calls
            high_volume_calls = calls[calls['volume'] > calls['volume'].quantile(0.8)]
            for _, option in high_volume_calls.iterrows():
                option_data = option.to_dict()
                option_data['type'] = 'call'  # Add type explicitly
                opportunities.append({
                    "type": "call",
                    "strike": float(option['strike']),
                    "expiration": expiration,
                    "volume": float(option['volume']),
                    "iv": float(option['impliedVolatility']),
                    "score": self._calculate_opportunity_score(option_data, technical)
                })
            
            # Process puts
            high_volume_puts = puts[puts['volume'] > puts['volume'].quantile(0.8)]
            for _, option in high_volume_puts.iterrows():
                option_data = option.to_dict()
                option_data['type'] = 'put'  # Add type explicitly
                opportunities.append({
                    "type": "put",
                    "strike": float(option['strike']),
                    "expiration": expiration,
                    "volume": float(option['volume']),
                    "iv": float(option['impliedVolatility']),
                    "score": self._calculate_opportunity_score(option_data, technical)
                })
        
        return {"opportunities": sorted(opportunities, key=lambda x: x["score"], reverse=True)}

    def _calculate_opportunity_score(self, option: pd.Series, technical: Dict[str, Any]) -> float:
        """Calculate a comprehensive score for each option opportunity"""
        score = 0.0
        
        # Volume score (0-30 points)
        volume = float(option['volume'])
        # Score based on volume thresholds
        if volume > 1000:
            score += 30
        elif volume > 500:
            score += 20
        elif volume > 100:
            score += 10
        
        # IV score (0-20 points)
        iv = float(option['impliedVolatility'])
        if 0.2 <= iv <= 0.4:
            score += 20
        elif 0.1 <= iv <= 0.5:
            score += 10
        
        # Strike price proximity score (0-20 points)
        current_price = technical.get('current_price', 0)
        if current_price > 0:
            strike = float(option['strike'])
            price_diff_pct = abs(strike - current_price) / current_price
            if price_diff_pct <= 0.05:  # within 5%
                score += 20
            elif price_diff_pct <= 0.10:  # within 10%
                score += 10
        
        # Technical signal alignment (0-30 points)
        if technical['macd_signal'] == "Buy" and option['type'] == "call":
            score += 15
        if technical['macd_signal'] == "Sell" and option['type'] == "put":
            score += 15
        if 30 <= technical['rsi'] <= 70:
            score += 15
            
        return score

    def _generate_recommendation(self, data: Dict[str, Any], options_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final trading recommendation"""
        best_opportunities = options_analysis["opportunities"][:3]
        
        return {
            "action": "BUY" if data["technical_signals"]["macd_signal"] == "Buy" else "SELL",
            "option_recommendations": best_opportunities,
            "reasoning": self._generate_reasoning(data),
            "risk_level": self._calculate_risk_level(data)
        }

    def _generate_reasoning(self, data: Dict[str, Any]) -> str:
        """Generate detailed reasoning for the recommendation"""
        technical = data['technical_signals']
        sentiment = data['news_sentiment']
        
        reasoning = []
        
        # Technical Analysis Reasoning
        reasoning.append("Technical Analysis:")
        if technical['rsi'] > 70:
            reasoning.append("- RSI indicates overbought conditions (>70)")
        elif technical['rsi'] < 30:
            reasoning.append("- RSI indicates oversold conditions (<30)")
        else:
            reasoning.append("- RSI is in neutral territory")
            
        reasoning.append(f"- MACD signals a {technical['macd_signal']} opportunity")
        reasoning.append(f"- Volatility is {technical['volatility']:.2f}")
        
        # Market Sentiment
        sentiment_level = "positive" if sentiment['sentiment_score'] > 0.6 else "negative" if sentiment['sentiment_score'] < 0.4 else "neutral"
        reasoning.append(f"\nMarket Sentiment:")
        reasoning.append(f"- News sentiment is {sentiment_level} ({sentiment['sentiment_score']:.2f})")
        
        # Options Analysis
        options_data = data['options_chain']
        if 'iv_skew' in options_data:
            skew = options_data['iv_skew']
            reasoning.append(f"\nOptions Analysis:")
            reasoning.append(f"- IV Skew: {skew:.2f} ({'bearish' if skew > 0.1 else 'bullish' if skew < -0.1 else 'neutral'})")
        
        return "\n".join(reasoning)

    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence level in the recommendation"""
        # Implement confidence calculation
        return 0.85

    def _calculate_risk_level(self, data: Dict[str, Any]) -> str:
        """Calculate risk level based on multiple factors"""
        technical = data['technical_signals']
        
        risk_score = 0
        
        # Volatility contribution
        vol_percentile = technical['volatility'] / technical.get('avg_volatility', technical['volatility'])
        if vol_percentile > 1.5:
            risk_score += 3  # High risk
        elif vol_percentile > 1.2:
            risk_score += 2  # Medium risk
        else:
            risk_score += 1  # Low risk
            
        # RSI contribution
        rsi = technical['rsi']
        if rsi > 75 or rsi < 25:
            risk_score += 3  # High risk
        elif rsi > 65 or rsi < 35:
            risk_score += 2  # Medium risk
        else:
            risk_score += 1  # Low risk
            
        # Final risk assessment
        if risk_score >= 5:
            return "High"
        elif risk_score >= 3:
            return "Medium"
        return "Low"

def run_options_analysis(ticker: str) -> str:
    """Run complete options analysis through the agent"""
    agent = StockAgent()
    result = agent.make_trading_decision(ticker)
    
    # Format the output
    output = f"""
    Options Trading Analysis for {ticker}
    
    Recommendation: {result['recommendation']['action']}
    Confidence: {result['confidence']*100:.1f}%
    
    Top Options Opportunities:
    """
    
    for opp in result['recommendation']['option_recommendations']:
        output += f"""
        {opp['type'].upper()} Option:
        - Strike: ${opp['strike']:.2f}
        - Expiration: {opp['expiration']}
        - Volume: {opp['volume']}
        - IV: {opp['iv']:.2f}
        - Score: {opp['score']:.2f}
        """
    
    output += f"""
    Technical Signals:
    - RSI: {result['supporting_data']['technical_signals']['rsi']:.2f}
    - MACD Signal: {result['supporting_data']['technical_signals']['macd_signal']}
    - Volatility: {result['supporting_data']['technical_signals']['volatility']:.2f}
    
    Market Sentiment:
    - News Sentiment Score: {result['supporting_data']['news_sentiment']['sentiment_score']:.2f}
    """
    
    return output

# Create Gradio interface
interface = gr.Interface(
    fn=run_options_analysis,
    inputs=gr.Textbox(label="Enter Stock Ticker (e.g., SPY, AAPL, NVDA)"),
    outputs=gr.Markdown(),
    title="AI Options Trading Analyst",
    description="Enter a stock ticker for comprehensive options trading analysis",
    examples=[["SPY"], ["AAPL"], ["NVDA"]]
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    interface.launch() 