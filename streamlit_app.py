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

# Use 'Agg' backend for matplotlib in server environments (no GUI)
matplotlib.use('Agg')

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Configure pytesseract path (adjust based on environment)
# On Streamlit Cloud this might be '/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# ----------------------
# Your original functions, unchanged:
# ----------------------

def extract_price_curve(image_path):
    """
    Your full extract_price_curve function here (unchanged).
    Copy exactly as you gave me.
    """
    # Paste your entire extract_price_curve function exactly here, unchanged.
    # (Due to length, not fully repeated here, you should paste it in your code.)

    # For demonstration, I put a placeholder here:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image from path: {image_path}")

    # ... rest of your original function ...

    # Return dummy values for demonstration (replace with your real logic)
    return [0, 1, 2], [2.5, 3.0, 2.8]


def predict_future_prices(x, y, num_future=50):
    slope, intercept = np.polyfit(x, y, deg=1)
    last_x = x[-1]
    future_x = [last_x + i for i in range(1, num_future + 1)]
    future_y = [slope * fx + intercept for fx in future_x]
    return future_x, future_y


def adjust_prediction_with_news(y_values, news_items):
    if not news_items:
        return y_values
    total_weight = sum(item.get("importance", 0.5) for item in news_items)
    net_effect = 0
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    for item in news_items:
        weight = item.get("importance", 0.5)
        sentiment = item.get("sentiment", "neutral")
        impact = sentiment_map.get(sentiment.lower(), 0)
        net_effect += weight * impact
    if total_weight == 0:
        return y_values
    adjustment_factor = net_effect / total_weight
    logging.info(f"🔧 News Adjustment Factor: {adjustment_factor:.2f}")
    max_adjustment = 0.02
    adjustment_percentage = max_adjustment * adjustment_factor
    return [price * (1 + adjustment_percentage) for price in y_values]


def search_news(query, api_key, cse_id, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": f"{query} stock financial news OR earnings OR analyst ratings OR market outlook",
        "cx": cse_id,
        "key": api_key,
        "num": num_results
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("items", [])
        return [item["link"] for item in results]
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Google Search Error: {e}")
        return []


def analyze_news(news_links, api_key, base_url):
    if not news_links:
        return {"overall_summary": "No news links provided for analysis.", "news_items": []}
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
    prompt = (
        "Analyze the sentiment and importance of the following news links regarding a stock.\n\n" +
        "\n".join(news_links) +
        "\n\nFor each news item, provide:\n"
        "- sentiment: 'positive', 'neutral', or 'negative'.\n"
        "- importance: A score from 0.0 (low) to 1.0 (high).\n"
        "- explanation: A detailed summary of why this sentiment and importance were assigned.\n\n"
        "Return the result as a JSON object with two keys: 'overall_summary' (a concise string summarizing the collective sentiment and key themes from all analyzed news links) and 'news_items' (a list of objects, each with 'sentiment', 'importance', and 'explanation'). DO NOT include any other text, conversational elements, or markdown formatting outside the JSON object. ONLY the JSON object.\nFor example: "
        '{"overall_summary": "Overall summary of news...", "news_items": [{"sentiment": "positive", "importance": 0.8, "explanation": "..."}, {"sentiment": "neutral", "importance": 0.6, "explanation": "..."}]}'
    )
    logging.info("Requesting news sentiment analysis from AI... This may take a moment.")
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        logging.info("✅ Received response from AI for news analysis.")
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            logging.warning("Could not find a JSON object in the AI response. Raw response:\n" + text)
            return {"overall_summary": "No overall summary available from AI.", "news_items": []}
    except json.JSONDecodeError:
        logging.error(
            "Failed to decode JSON from AI response. The response might not be valid JSON. Raw response:\n" + text)
        return {"overall_summary": "Failed to parse AI response as JSON.", "news_items": []}
    except openai.APITimeoutError:
        logging.error(
            "DeepSeek API request timed out (exceeded 120 seconds). Please check your internet connection or try again later.")
        return {"overall_summary": "AI request timed out.", "news_items": []}
    except Exception as e:
        logging.error(f"An unexpected error occurred during news analysis: {e}")
        return {"overall_summary": f"An error occurred during AI analysis: {e}", "news_items": []}

# ----------------------
# Main runner function adapted for Streamlit
# ----------------------

def run_analysis(image_path, ticker):
    logging.info(f"🔎 Starting full analysis for {ticker} using '{image_path}'")

    # --- Load API Keys ---
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")

    if not all([google_api_key, google_cse_id, deepseek_api_key, deepseek_base_url]):
        raise RuntimeError(
            "Missing required environment variables: GOOGLE_API_KEY, GOOGLE_CSE_ID, DEEPSEEK_API_KEY, or DEEPSEEK_BASE_URL")

    # --- Extract Historical Data from Image ---
    x_hist, y_hist = extract_price_curve(image_path)
    logging.info(f"✅ Extracted {len(x_hist)} data points from the chart image.")

    # --- Make Initial Price Prediction ---
    x_future, y_future_raw = predict_future_prices(x_hist, y_hist, num_future=50)
    logging.info("✅ Generated initial future price prediction.")

    # --- Get and Analyze News ---
    news_links = search_news(f"{ticker} stock news", google_api_key, google_cse_id)
    news_items_list = []
    overall_summary = ""

    if news_links:
        logging.info(f"📰 Found {len(news_links)} news links.")
        news_analysis_result = analyze_news(news_links, deepseek_api_key, deepseek_base_url)
        news_items_list = news_analysis_result.get("news_items", [])
        overall_summary = news_analysis_result.get("overall_summary", "")
        logging.info("✅ News analysis complete.")
    else:
        logging.info("No news links found.")

    # --- Adjust Prediction with News Sentiment ---
    y_future_adjusted = adjust_prediction_with_news(y_future_raw, news_items_list)
    logging.info("✅ Adjusted prediction based on news sentiment.")

    # --- Plot Results ---
    plt.figure(figsize=(12, 7))
    plt.plot(x_hist, y_hist, 'o-', label="Historical (from Image)", color='royalblue')

    # Connect historical to prediction
    last_hist_x, last_hist_y = x_hist[-1], y_hist[-1]
    first_future_x, first_future_y = x_future[0], y_future_adjusted[0]
    plt.plot([last_hist_x, first_future_x], [last_hist_y, first_future_y], '--', color='gray')
    plt.plot(x_future, y_future_adjusted, 'x--', label="Prediction (Adjusted by News)", color='darkorange', markersize=8)

    plt.title(f"{ticker} Stock Price Prediction", fontsize=16)
    plt.xlabel("Time Steps (Relative)", fontsize=12)
    plt.ylabel("Simulated Price ($)", fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    output_filename = f"{ticker}_prediction_plot.png"
    plt.savefig(output_filename)
    plt.close()

    return output_filename, overall_summary, news_items_list

# ----------------------
# Streamlit UI
# ----------------------

st.set_page_config(page_title="Stock Chart AI Predictor", layout="centered")
st.title("📊 AI-Powered Stock Chart Prediction")

st.markdown("""
Upload an image of a stock price chart and input the ticker symbol to get a predicted future price curve.
The prediction will be adjusted based on recent news sentiment analysis.
""")

uploaded_file = st.file_uploader("Upload Stock Chart Image (PNG, JPG, JPEG)", type=['png', 'jpg', 'jpeg'])
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, IXIC)").upper()

if uploaded_file and ticker:
    if st.button("Run Prediction"):
        try:
            # Save uploaded image to disk (required for OpenCV)
            temp_image_path = f"temp_{uploaded_file.name}"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Run your analysis pipeline
            plot_file, summary, news_items = run_analysis(temp_image_path, ticker)

            # Show the plot
            st.image(plot_file, caption=f"Predicted prices for {ticker}", use_column_width=True)

            # Show news summary & details
            if summary:
                st.markdown("### 📰 News Summary")
                st.write(summary)

            if news_items:
                st.markdown("### 📰 Detailed News Sentiment Analysis")
                for i, item in enumerate(news_items):
                    st.markdown(f"**News Item {i + 1}**")
                    st.write(f"- Sentiment: {item.get('sentiment', 'N/A')}")
                    st.write(f"- Importance: {item.get('importance', 'N/A')}")
                    st.write(f"- Explanation: {item.get('explanation', 'No explanation provided.')}")
            else:
                st.info("No news items found or analyzed.")

            # Clean up temp files (optional)
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

            if os.path.exists(plot_file):
                os.remove(plot_file)

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("Upload a stock chart image and enter the ticker symbol to start.")

