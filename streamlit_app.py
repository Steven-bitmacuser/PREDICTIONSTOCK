import os
import logging
import json
import re
import cv2
import numpy as np
import requests
import openai
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import pytesseract

# Matplotlib backend for server environments
matplotlib.use('Agg')

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)

# Configure pytesseract path (may need adjusting for local/non-local)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # For Streamlit Cloud

# ========== YOUR EXISTING FUNCTIONS (unchanged) ==========
# Include your original `extract_price_curve`, `predict_future_prices`,
# `adjust_prediction_with_news`, `search_news`, `analyze_news`, etc.
# I won’t repeat them here, just keep your originals as-is.
# ==========================================================

# --- run_analysis (unchanged core, modified to return filename and result for Streamlit display) ---
def run_analysis(image_path, ticker):
    logging.info(f"🔎 Starting full analysis for {ticker} using '{image_path}'")

    # Load API keys
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")

    if not all([google_api_key, google_cse_id, deepseek_api_key, deepseek_base_url]):
        raise ValueError("Missing required environment variables (API keys). Check your .env file.")

    # 1. Extract Data from Image
    x_hist, y_hist = extract_price_curve(image_path)

    # 2. Predict Future Prices
    x_future, y_future_raw = predict_future_prices(x_hist, y_hist, num_future=50)

    # 3. News
    news_links = search_news(f"{ticker} stock news", google_api_key, google_cse_id)
    news_items = []

    if news_links:
        news_analysis = analyze_news(news_links, deepseek_api_key, deepseek_base_url)
        news_items = news_analysis.get("news_items", [])

    # 4. Adjust Prediction
    y_future_adjusted = adjust_prediction_with_news(y_future_raw, news_items)

    # 5. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_hist, y_hist, 'o-', label="Historical (Image)", color='royalblue')
    plt.plot(x_future, y_future_adjusted, 'x--', label="Prediction (News Adjusted)", color='darkorange')
    plt.grid(True)
    plt.legend()
    plt.title(f"{ticker} Stock Forecast")
    plt.xlabel("Time")
    plt.ylabel("Price ($)")
    plt.tight_layout()

    output_file = f"{ticker}_prediction_plot.png"
    plt.savefig(output_file)
    plt.close()

    return output_file, news_items

# === STREAMLIT APP UI ===
st.set_page_config(page_title="Stock Prediction AI", layout="centered")
st.title("📈 Stock Chart Predictor with AI + News")

st.markdown("""
Upload a stock chart screenshot (like from WeChat, Yahoo Finance) and enter the stock ticker (like AAPL, TSLA, IXIC).
The AI will extract price points from the image, predict future prices, and adjust them based on recent news sentiment.
""")

# Upload chart
uploaded_file = st.file_uploader("📤 Upload Stock Chart (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

# Ticker input
ticker = st.text_input("💡 Enter Stock Ticker Symbol", placeholder="e.g., AAPL, TSLA, IXIC")

# Run analysis
if uploaded_file and ticker:
    if st.button("🚀 Predict Stock Price"):
        with st.spinner("Analyzing chart, predicting prices, and fetching news..."):
            try:
                image_path = f"uploaded_chart.{uploaded_file.name.split('.')[-1]}"
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Run core logic
                result_img, news_result = run_analysis(image_path, ticker)

                st.success("✅ Analysis complete!")
                st.image(result_img, caption="📉 Prediction Result", use_column_width=True)

                if news_result:
                    st.markdown("### 📰 News Impact on Prediction")
                    for i, item in enumerate(news_result):
                        st.markdown(f"**News {i + 1}**")
                        st.write(f"- Sentiment: **{item['sentiment']}**")
                        st.write(f"- Importance: `{item['importance']}`")
                        st.write(f"- Explanation: {item['explanation']}")
                else:
                    st.info("No news data was used in the prediction.")

            except Exception as e:
                st.error(f"❌ Error: {e}")
else:
    st.info("👆 Upload an image and enter a ticker to begin.")

