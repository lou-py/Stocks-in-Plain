# Tracker 3 datas en utilisant un LLM pour résumer la situation

# 1 · Imports
import streamlit as st
import yfinance as yf
import pandas as pd
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# 2 · Config
TICKERS = ["ASML","TSM", "NVDA"]
RSS_URL = "https://news.google.com/rss/search?q={ticker}+stock"

# petit modèle de génération 
summariser = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device_map="auto",          
)

# 3 · Fonctions utilitaires
def get_price_data(ticker: str) -> pd.DataFrame:
    """1 an de prix quotidien."""
    return yf.download(ticker, period="1y", interval="1d", progress=False)

def basic_metrics(df: pd.DataFrame):
    """Prix, move jour, YTD et vol 30 j (en %)."""
    latest   = df["Close"].iloc[-1]
    daily_px = df["Close"].pct_change().iloc[-1] * 100
    ytd_px   = (latest / df["Close"].iloc[0] - 1) * 100
    vol_30d  = df["Close"].pct_change().rolling(30).std().iloc[-1] * (100 ** 0.5)
    return round(latest, 2), round(daily_px, 2), round(ytd_px, 2), round(vol_30d, 2)

def get_headlines(ticker: str, n: int = 3):
    feed = feedparser.parse(RSS_URL.format(ticker=ticker))
    return [e.title for e in feed.entries[:n]]

def score_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    return sum(scores) / len(scores) if scores else 0.0

def llm_summary(ticker: str, metrics, senti):
    """
    Retourne une phrase (≤ 25 mots) résumant la situation.
    Si le LLM échoue, on construit une phrase simple.
    """
    sentiment = (
        "positive" if senti > 0.2 else
        "negative" if senti < -0.2 else
        "neutral"
    )
    prompt = (
        f"Write one plain English sentence (max 25 words) : "
        f"{ticker} trades at ${float(metrics[0]):.2f}, "
        f"{float(metrics[1]):+.2f}% today, {float(metrics[2]):+.2f}% YTD, "
        f"30-day vol {float(metrics[3]):.2f}%, news sentiment is {sentiment}."
    )

    try:
        out = summariser(prompt, max_length=40, num_beams=4)[0]["generated_text"]
        return out.strip().rstrip(".") + "."
    except Exception as e:
        # secours rudimentaire
        return (
            f"{ticker} is {metrics[1]:+.2f}% today and "
            f"{metrics[2]:+.2f}% YTD; news tone looks {sentiment}."
        )

# 4 · Interface Streamlit
st.set_page_config(page_title="Stocks in Plain English", layout="wide")
st.title("📈 Stocks-in-Plain-English Dashboard")

for tic in TICKERS:
    st.header(tic)

    # --- données & calculs
    df = get_price_data(tic)
    price, day_ret, ytd_ret, vol30 = map(lambda x: x.iloc[-1] if isinstance(x, pd.Series) else x, basic_metrics(df))
    headlines = get_headlines(tic)
    senti = score_sentiment(headlines)
    summary = llm_summary(tic, (price, day_ret, ytd_ret, vol30), senti)

    # --- mise en page
    col1, col2 = st.columns([2, 1])

    with col1:
        st.line_chart(df["Close"][-90:])

    with col2:
        st.metric("Latest Price", f"${price:,.2f}")
        st.metric("Today", f"{day_ret:+.2f} %")
        st.metric("YTD", f"{ytd_ret:+.2f} %")
        st.metric("30-Day Vol", f"{vol30:.2f} %")
        st.progress((senti + 1) / 2)

    st.write("**News Headlines**")
    for h in headlines:
        st.write("•", h)

    st.info(summary)
    st.divider()
