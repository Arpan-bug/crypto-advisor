import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# App setup
st.set_page_config(page_title="Crypto Advisor", layout="wide")
st.title("🧠 Crypto Advisor Tool")
st.subheader("Smart signals for high-potential coins")

# Yahoo-compatible coin map
yahoo_map = {
    "Solana": "SOL-USD",
    "XRP": "XRP-USD",
    "Chainlink": "LINK-USD",
    "Algorand": "ALGO-USD",
    "Cartesi": "CTSI-USD",
    "AMP": "AMP-USD",
    "SUI": "SUI1-USD",
    "Powerledger": "POWR-USD",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
}

# Sidebar
selected_coin = st.sidebar.selectbox("Select a coin", list(yahoo_map.keys()))
symbol = yahoo_map[selected_coin]

# Price data
df = yf.download(symbol, period="90d", interval="1d")
if df.empty or "Close" not in df.columns:
    st.error("❌ Failed to fetch price data. Try another coin.")
    st.stop()

df = df[["Close"]].rename(columns={"Close": "price"})
df["SMA20"] = df["price"].rolling(window=20).mean()

# RSI
delta = df["price"].diff()
gain = np.where(delta > 0, delta, 0).flatten()
loss = np.where(delta < 0, -delta, 0).flatten()
avg_gain = pd.Series(gain, index=df.index).rolling(window=14).mean()
avg_loss = pd.Series(loss, index=df.index).rolling(window=14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# MACD
ema12 = df["price"].ewm(span=12, adjust=False).mean()
ema26 = df["price"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Histogram"] = df["MACD"] - df["Signal"]

# Chart: Price + SMA
st.subheader(f"📈 {selected_coin} – Price & SMA")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["price"], label="Price", linewidth=2)
ax.plot(df.index, df["SMA20"], label="SMA 20", linestyle="--")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# Show Price + RSI
latest_price = float(df["price"].iloc[-1])
st.markdown(f"### 🧪 Current Price: **${latest_price:.4f}**")

if df["RSI"].dropna().empty:
    st.warning("⚠️ Not enough data to calculate RSI.")
else:
    latest_rsi = float(df["RSI"].dropna().iloc[-1])
    st.markdown(f"### 📊 RSI: **{latest_rsi:.2f}**")
    if latest_rsi < 30:
        st.success("✅ RSI suggests **oversold**. Possible entry zone.")
    elif latest_rsi > 70:
        st.warning("⚠️ RSI suggests **overbought**. Consider caution.")
    else:
        st.info("ℹ️ RSI is neutral. No strong signal.")

# MACD Chart
st.subheader("📉 MACD Indicator")
fig_macd, ax_macd = plt.subplots(figsize=(10, 3))
ax_macd.plot(df.index, df["MACD"], label="MACD", color="blue")
ax_macd.plot(df.index, df["Signal"], label="Signal", color="orange")
ax_macd.bar(df.index, df["Histogram"], label="Histogram", color="gray")
ax_macd.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax_macd.legend()
st.pyplot(fig_macd)

# MACD advice
if df["MACD"].notna().iloc[-1] and df["Signal"].notna().iloc[-1]:
    macd_now = df["MACD"].iloc[-1]
    macd_prev = df["MACD"].iloc[-2]
    signal_now = df["Signal"].iloc[-1]
    signal_prev = df["Signal"].iloc[-2]

    if macd_prev < signal_prev and macd_now > signal_now:
        st.success("🚀 MACD Bullish Crossover! Momentum may be shifting upward.")
    elif macd_prev > signal_prev and macd_now < signal_now:
        st.warning("🔻 MACD Bearish Crossover! Possible downward shift.")
    else:
        st.info("📊 MACD is neutral. No strong signal currently.")

# =======================
# NEWS SENTIMENT ANALYSIS
# =======================

st.subheader(f"📰 {selected_coin} — Live News Sentiment")

rss_url = "https://feeds.feedburner.com/CoinDesk"
feed = feedparser.parse(rss_url)

# Match article title with coin name or symbol
coin_keywords = [selected_coin.lower()]
symbol_guess = symbol.split("-")[0].lower()
if symbol_guess != selected_coin.lower():
    coin_keywords.append(symbol_guess)

filtered = [
    entry for entry in feed.entries
    if any(k in entry.title.lower() for k in coin_keywords)
]

if not filtered:
    st.warning(f"No headlines matched **{selected_coin}**. Showing general crypto news.")
    filtered = feed.entries[:6]

# Analyze sentiment with VADER
analyzer = SentimentIntensityAnalyzer()
emoji_map = {"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴"}
sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
headline_scores = []

for entry in filtered:
    title = entry.title
    polarity = analyzer.polarity_scores(title)["compound"]

    if polarity > 0.25:
        sentiment = "Positive"
    elif polarity < -0.25:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    sentiment_counts[sentiment] += 1
    emoji = emoji_map[sentiment]
    headline_scores.append((emoji, title, sentiment, polarity))

# Show headlines sorted by sentiment strength
headline_scores.sort(key=lambda x: abs(x[3]), reverse=True)

for emoji, title, sentiment, polarity in headline_scores:
    st.markdown(f"{emoji} **{title}** — *{sentiment}*")

# Sentiment trend chart
st.markdown("### 📊 Sentiment Trend")
if len(headline_scores) > 2:
    trend_df = pd.DataFrame({
        "Headline": [h[1] for h in headline_scores],
        "Polarity": [h[3] for h in headline_scores]
    }).sort_values(by="Polarity", key=lambda x: x.abs(), ascending=False)

    fig_sent, ax_sent = plt.subplots(figsize=(8, 2.5))
    ax_sent.plot(trend_df["Polarity"], marker="o", linestyle="-")
    ax_sent.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax_sent.set_title("Headline Sentiment Strength")
    ax_sent.set_xlabel("Headline")
    ax_sent.set_ylabel("Score")
    st.pyplot(fig_sent)

# Final summary
st.divider()
st.markdown("### 🧠 Final Sentiment Insight")
total = sum(sentiment_counts.values())

if total >= 5:
    net = sentiment_counts["Positive"] - sentiment_counts["Negative"]
    bias_ratio = abs(net) / total

    if net > 0 and bias_ratio >= 0.6:
        st.success("📈 Strong bullish bias detected in news.")
    elif net < 0 and bias_ratio >= 0.6:
        st.error("📉 Strong bearish bias detected in news.")
    else:
        st.info("📊 News sentiment is mixed or weak — no clear signal.")
else:
    st.warning("⚠️ Not enough relevant headlines for confident news-based advice.")
# ============================
# 🎯 Confidence Score Section
# ============================

st.divider()
st.subheader("🎯 Combined Signal Confidence Score")

confidence_score = 0
details = []

# RSI score
if not df["RSI"].dropna().empty:
    rsi_latest = df["RSI"].dropna().iloc[-1]
    if rsi_latest < 30:
        confidence_score += 1
        details.append("🟢 RSI suggests Oversold")
    elif rsi_latest > 70:
        confidence_score -= 1
        details.append("🔴 RSI suggests Overbought")
    else:
        details.append("⚪ RSI neutral")

# MACD score
macd_valid = df["MACD"].notna().iloc[-1] and df["Signal"].notna().iloc[-1]
if macd_valid:
    macd_now = df["MACD"].iloc[-1]
    macd_prev = df["MACD"].iloc[-2]
    signal_now = df["Signal"].iloc[-1]
    signal_prev = df["Signal"].iloc[-2]

    if macd_prev < signal_prev and macd_now > signal_now:
        confidence_score += 2
        details.append("🟢 MACD Bullish Crossover")
    elif macd_prev > signal_prev and macd_now < signal_now:
        confidence_score -= 2
        details.append("🔴 MACD Bearish Crossover")
    else:
        details.append("⚪ MACD Neutral")

# News sentiment score
if total >= 5:
    net_news = sentiment_counts["Positive"] - sentiment_counts["Negative"]
    ratio = abs(net_news) / total

    if net_news > 0 and ratio >= 0.6:
        confidence_score += 2
        details.append("🟢 Bullish News Sentiment")
    elif net_news < 0 and ratio >= 0.6:
        confidence_score -= 2
        details.append("🔴 Bearish News Sentiment")
    else:
        details.append("⚪ News Sentiment Mixed")

# Display score details
for d in details:
    st.markdown(d)

# Final signal
st.markdown("### 🧾 Final Score:")
if confidence_score >= 2:
    st.success(f"🟢 High Confidence Signal (Score: {confidence_score})")
elif confidence_score <= -2:
    st.error(f"🔴 Low Confidence Signal (Score: {confidence_score})")
else:
    st.info(f"⚪ Neutral Signal (Score: {confidence_score})")
