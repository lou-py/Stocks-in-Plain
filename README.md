# Stocks-in-Plain
My side project that mixes market data, sentiment analysis and a tiny open-source LLM. Built with Streamlit so anyone can poke around without installing heavy stuff.

I’m a first–year finance student learning Python. I wanted one mini‑project that touches three buzz‑words every internship posting mentions:
Market data (Yahoo Finance)
Sentiment / NLP (VADER)
Generative AI (Transformers)
Plus, I love semiconductors so I picked ASML, TSM, NVDA.

# How it works ?

Download prices – yfinance pulls 1‑year daily closes.
Calc basic stats – daily % move, YTD %, 30‑day vol.
Grab news – Google News RSS, keep the top 3 titles.
Sentiment – average VADER compound score.
LLM summary – feed the numbers + sentiment into google/flan‑t5-small ➜ 1 sentence.

Everything is local & free

# What I learned

Reading price data into pandas and doing rolling calculations.
Basic sentiment scoring with VADER.
Loading a tiny open‑source LLM and creating a prompt.
Building a fast UI with Streamlit (columns, metrics, progress bar).

# Possible next steps

Switch to intraday (1‑min) prices for more action.
Add more tickers + a selectbox.
Push to Streamlit Community Cloud so friends can click the link.
Compare FLAN vs. GPT(pay‑vs‑free trade‑off).

