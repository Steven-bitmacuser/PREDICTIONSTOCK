import streamlit as st
import os
import argparse
import logging
import json
import re
import cv2
import numpy as np
import requests
import openai
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import shutil
import pandas as pd
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import pytz

# Set up logging for Streamlit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()  # Load environment variables from .env file

# --- Configure pytesseract ---
try:
    # Attempt to find tesseract in the system's PATH first
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        logging.info(f"Pytesseract command set to: {tesseract_path}")
    else:
        # Fallback to a common Linux path if not found in PATH
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
        logging.warning(f"Tesseract not found in system PATH. Attempting fallback to: {pytesseract.pytesseract.tesseract_cmd}")

    # Verify if tesseract_cmd is set and tesseract is accessible
    if not pytesseract.pytesseract.tesseract_cmd:
        raise FileNotFoundError("Tesseract executable path not found or set.")
    
    # Optional: Verify Tesseract version to ensure it's callable
    pytesseract.get_tesseract_version() 

except pytesseract.TesseractNotFoundError:
    st.error(
        "❌ Tesseract OCR engine not found! Please ensure Tesseract is installed on your system or configured "
        "correctly in your environment variables/`packages.txt` for Streamlit Cloud deployment. "
        "Refer to the Streamlit documentation for deploying apps with system dependencies."
    )
    st.stop()
except FileNotFoundError as e:
    st.error(
        f"❌ Pytesseract configuration error: {e}. The Tesseract executable path could not be determined. "
        "Please ensure Tesseract is installed and its path is correctly set."
    )
    st.stop()
except Exception as e:
    st.error(
        f"❌ An unexpected error occurred during Pytesseract configuration: {e}. "
        "Please check your setup."
    )
    st.stop()


# --- chart_extractor.py content (Enhanced for Maximum Data Points) ---
def extract_price_curve(image):
    """
    Extracts a price curve and its real values from a chart image using computer vision and OCR.

    This version is highly optimized for extracting a large number of data points
    from a distinctively colored stock price line (like red/pink in WechatIMG204 2.jpg).
    """
    if image is None:
        raise ValueError("Image object is None.")

    # Convert PIL Image to OpenCV format
    image_np = np.array(image)
    # For some images, RGB to BGR conversion might be needed
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif image_np.ndim == 2:  # Grayscale image
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    else:
        image_cv = image_np  # Assume already BGR or compatible

    h, w, _ = image_cv.shape

    # --- 1. OCR for Y-Axis Labels (Universal Detection - unchanged) ---
    left_y_axis_roi_x_start = 0
    left_y_axis_roi_x_end = int(w * 0.25)
    right_y_axis_roi_x_start = int(w * 0.75)
    right_y_axis_roi_x_end = w
    y_axis_roi_y_start = int(h * 0.1)
    y_axis_roi_y_end = int(h * 0.95)

    potential_y_axes_data = []

    left_roi = image_cv[y_axis_roi_y_start:y_axis_roi_y_end, left_y_axis_roi_x_start:left_y_axis_roi_x_end]
    if left_roi.size > 0:
        gray_left_roi = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
        inverted_left_roi = cv2.bitwise_not(gray_left_roi)
        _, ocr_thresh_left = cv2.threshold(inverted_left_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ocr_data_left = pytesseract.image_to_data(ocr_thresh_left, output_type=pytesseract.Output.DICT,
                                                     config='--psm 6')
        potential_y_axes_data.append((ocr_data_left, y_axis_roi_y_start, "left"))

    right_roi = image_cv[y_axis_roi_y_start:y_axis_roi_y_end, right_y_axis_roi_x_start:right_y_axis_roi_x_end]
    if right_roi.size > 0:
        gray_right_roi = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)
        inverted_right_roi = cv2.bitwise_not(gray_right_roi)
        _, ocr_thresh_right = cv2.threshold(inverted_right_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ocr_data_right = pytesseract.image_to_data(ocr_thresh_right, output_type=pytesseract.Output.DICT,
                                                     config='--psm 6')
        potential_y_axes_data.append((ocr_data_right, y_axis_roi_y_start, "right"))

    y_axis_labels = []
    selected_axis_side = None

    best_label_count = 0
    for ocr_data, roi_y_offset, side in potential_y_axes_data:
        current_side_labels = []
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            if not text or len(text) < 2:
                continue
            if re.fullmatch(r"^\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?$", text.replace(",", "")):
                try:
                    value = float(text.replace(",", ""))
                    y_center_pixel = ocr_data['top'][i] + ocr_data['height'][i] // 2 + roi_y_offset
                    current_side_labels.append({'value': value, 'y_pixel': y_center_pixel})
                except ValueError:
                    continue

        if len(current_side_labels) > best_label_count:
            y_axis_labels = current_side_labels
            best_label_count = len(current_side_labels)
            selected_axis_side = side

    if not y_axis_labels:
        logging.error("No numeric Y-axis labels found via OCR on either side. Cannot determine price scale.")
        min_price_val = 2.6  # Fallback to a range suitable for WechatIMG204 2.jpg
        max_price_val = 3.8
        min_price_pixel = h * 0.9  # Assume lowest pixel is 90% down
        max_price_pixel = h * 0.1  # Assume highest pixel is 10% down
        logging.warning(f"Falling back to simulated range: {min_price_val}-{max_price_val} due to OCR failure.")
    else:
        y_axis_labels.sort(key=lambda x: x['y_pixel'])
        min_price_val = y_axis_labels[-1]['value']
        min_price_pixel = y_axis_labels[-1]['y_pixel']
        max_price_val = y_axis_labels[0]['value']
        max_price_pixel = y_axis_labels[0]['y_pixel']

        if min_price_pixel == max_price_pixel or min_price_val == max_price_val:
            logging.warning("Insufficient distinct Y-axis labels for precise scaling. Falling back to simulated range.")
            min_price_val = 2.6  # Fallback to a range suitable for WechatIMG204 2.jpg
            max_price_val = 3.8
            min_price_pixel = h * 0.9
            max_price_pixel = h * 0.1

    logging.info(
        f"OCR detected Y-axis ({selected_axis_side}): Min Price: {min_price_val} at Y-pixel {min_price_pixel}, Max Price: {max_price_val} at Y-pixel {max_price_pixel}")

    # --- 2. Extract Price Line (Optimized for distinct color lines) ---

    # Define a tighter ROI for the chart area to exclude grid lines and other noise
    # These values are tuned for WechatIMG204 2.jpg. Adjust for other images.
    chart_roi_x_start = int(w * 0.15)  # Start after left Y-axis labels
    chart_roi_x_end = int(w * 0.85)  # End before right edge
    chart_roi_y_start = int(h * 0.4)  # Start below top header/grid lines
    chart_roi_y_end = int(h * 0.85)  # End above bottom time labels/grid lines

    chart_area = image_cv[chart_roi_y_start:chart_roi_y_end, chart_roi_x_start:chart_roi_x_end]

    if chart_area.size == 0:
        logging.error("Chart ROI is empty. Check image dimensions or ROI coordinates.")
        raise ValueError("Chart ROI is empty, cannot perform line detection.")

    # Convert to HSV color space for robust color detection
    hsv = cv2.cvtColor(chart_area, cv2.COLOR_BGR2HSV)

    # Define a more precise range for the distinct red/pink line in WechatIMG204 2.jpg
    # These values were found by sampling the line color in the image.
    # Hue (H): 0-179 (red is around 0 and 170-180)
    # Saturation (S): 0-255 (how vibrant the color is)
    # Value (V): 0-255 (how bright the color is)
    lower_line_color1 = np.array([0, 150, 100])  # Strong red
    upper_line_color1 = np.array([10, 255, 255])

    lower_line_color2 = np.array([170, 150, 100])  # Other red range
    upper_line_color2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_line_color1, upper_line_color1)
    mask2 = cv2.inRange(hsv, lower_line_color2, upper_line_color2)
    line_mask = mask1 + mask2

    # Morphological operations to clean the mask and make the line continuous
    # Use a small kernel to connect tiny gaps without distorting the line shape too much
    kernel = np.ones((2, 2), np.uint8)  # Smaller kernel for finer detail
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours on the cleaned mask
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback to general edge detection if color-based fails, but it's less precise for dense points
        logging.warning("No distinct colored line contours found. Falling back to general edge detection.")
        gray_chart_area = cv2.cvtColor(chart_area, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray_chart_area)
        blurred_chart_area = cv2.GaussianBlur(contrast_enhanced, (3, 3), 0)  # Smaller blur for more detail
        edges = cv2.Canny(blurred_chart_area, 20, 80)  # Lower thresholds for more edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found in the chart area after color and fallback edge detection.")

    all_line_points = []
    # Collect all points from all significant contours
    for contour in contours:
        # Filter out very small noise contours, but keep enough detail
        if cv2.contourArea(contour) > 1:  # Very low threshold to capture almost all line pixels
            for pt in contour:
                # Adjust points to original image coordinates by adding ROI offsets
                all_line_points.append((pt[0][0] + chart_roi_x_start, pt[0][1] + chart_roi_y_start))

    if not all_line_points:
        raise ValueError("No valid points extracted from any significant contour.")

    # Sort points by the x-coordinate and aggregate, taking the median y for robustness
    # This ensures a single, representative Y-value for each X-pixel, maximizing density.
    points_by_x = {}
    for x, y in all_line_points:
        if x not in points_by_x:
            points_by_x[x] = []
        points_by_x[x].append(y)

    cleaned_points = []
    for x_coord in sorted(points_by_x.keys()):
        median_y = int(np.median(points_by_x[x_coord]))
        cleaned_points.append((x_coord, median_y))

    if not cleaned_points:
        raise ValueError("No valid points extracted from the main line after cleaning.")

    x_pixel_vals = [p[0] for p in cleaned_points]
    y_pixel_vals = [p[1] for p in cleaned_points]

    # --- 3. Map Pixel Y-values to Real Price Values (unchanged) ---
    y_vals_real = np.interp(y_pixel_vals,
                            (max_price_pixel, min_price_pixel),
                            (max_price_val, min_price_val)
                            ).tolist()

    x_vals = list(range(len(x_pixel_vals)))

    return x_vals, y_vals_real


# --- prediction.py content (unchanged, num_future already increased) ---
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


# --- news_service.py content (unchanged) ---
def search_news(query, api_key, cse_id, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": f"{query} stock financial news OR earnings OR analyst ratings OR market outlook", "cx": cse_id,
              "key": api_key, "num": num_results}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("items", [])
        return [item["link"] for item in results]
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Google Search Error: {e}")
        st.error(f"Google Search Error: {e}. Please check your GOOGLE_API_KEY and GOOGLE_CSE_ID.")
        return []


def analyze_news(news_links, api_key, base_url):
    if not news_links:
        return {"overall_summary": "No news links provided for analysis.", "news_items": []}
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
    prompt = (
            "Analyze the sentiment and importance of the following news links regarding a stock.\\n\\n" +
            "\\n".join(news_links) +
            "\\n\\nFor each news item, provide:\\n"
            "- sentiment: 'positive', 'neutral', or 'negative'.\\n"
            "- importance: A score from 0.0 (low) to 1.0 (high).\\n"
            "- explanation: A detailed summary of why this sentiment and importance were assigned.\\n\\n"
            "Return the result as a JSON object with two keys: 'overall_summary' (a concise string summarizing the collective sentiment and key themes from all analyzed news links) and 'news_items' (a list of objects, each with 'sentiment', 'importance', and 'explanation'). DO NOT include any other text, conversational elements, or markdown formatting outside the JSON object. ONLY the JSON object.\\nFor example: "
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
        st.error("Failed to decode JSON from AI response. The response might not be valid JSON.")
        return {"overall_summary": "Failed to parse AI response as JSON.", "news_items": []}
    except openai.APITimeoutError:
        logging.error(
            "DeepSeek API request timed out (exceeded 120 seconds). Please check your internet connection or try again later.")
        st.error("AI request timed out. Please check your internet connection or try again later.")
        return {"overall_summary": "AI request timed out.", "news_items": []}
    except Exception as e:
        logging.error(f"An unexpected error occurred during news analysis: {e}")
        st.error(f"An unexpected error occurred during AI analysis: {e}")
        return {"overall_summary": f"An error occurred during AI analysis: {e}", "news_items": []}


# --- main.py content (adapted for Streamlit) ---
def run_analysis_streamlit(uploaded_file, ticker):
    """
    Runs the full stock analysis workflow for Streamlit.
    """
    st.info(f"🔎 Starting full analysis for {ticker} using the uploaded image.")

    # --- 1. Load API Keys ---
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")

    if not all([google_api_key, google_cse_id, deepseek_api_key, deepseek_base_url]):
        st.error(
            "Missing one or more required environment variables. Please check your `.env` file or Streamlit secrets.")
        st.markdown("""
        **Required Environment Variables for Predictor:**
        - `GOOGLE_API_KEY`
        - `GOOGLE_CSE_ID`
        - `DEEPSEEK_API_KEY`
        - `DEEPSEEK_BASE_URL` (e.g., `https://api.deepseek.com`)
        - `TESSERACT_CMD` (Optional, if tesseract is not in your system's PATH, though auto-detection is preferred)
        """)
        return

    # Convert uploaded file to PIL Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Stock Chart", use_column_width=True)

    # --- 2. Extract Historical Data from Image ---
    with st.spinner("Extracting historical data from image..."):
        try:
            x_hist, y_hist = extract_price_curve(image)
            st.success(f"✅ Extracted {len(x_hist)} data points from the chart image.")
        except (FileNotFoundError, ValueError) as e:
            st.error(f"❌ Failed to extract data from image: {e}")
            return

    # --- 3. Make Initial Price Prediction ---
    with st.spinner("Generating initial price prediction..."):
        x_future, y_future_raw = predict_future_prices(x_hist, y_hist, num_future=50)
        st.success("✅ Generated initial future price prediction.")

    # --- 4. Get and Analyze News ---
    news_analysis = []
    overall_news_summary = "No news analysis performed."
    with st.spinner("Searching and analyzing news... This may take a while."):
        news_links = search_news(f"{ticker} stock news", google_api_key, google_cse_id)
        if news_links:
            st.subheader("📰 Found News Articles:")
            for link in news_links:
                st.write(f"- [{link}]({link})")

            news_analysis_result = analyze_news(news_links, deepseek_api_key, deepseek_base_url)

            news_items_list = news_analysis_result.get('news_items', [])
            overall_news_summary = news_analysis_result.get('overall_summary', 'No overall summary provided.')

            if news_items_list:
                st.success("✅ News analysis complete.")
                st.subheader("Overall News Summary:")
                st.write(overall_news_summary)
                st.subheader("Individual News Analysis Details:")
                for i, item in enumerate(news_items_list):
                    st.markdown(f"**News Item {i + 1}:**")
                    st.write(f"  **Sentiment:** {item.get('sentiment', 'N/A')}")
                    st.write(f"  **Importance:** {item.get('importance', 'N/A'):.2f}")
                    st.write(f"  **Explanation:** {item.get('explanation', 'No explanation provided.')}")
                news_analysis = news_items_list
            else:
                st.warning(
                    "News analysis returned no results or encountered an error. Prediction will not be adjusted by news.")
        else:
            st.warning("Could not find any news articles. Prediction will not be adjusted by news.")

    # --- 5. Adjust Prediction with News Sentiment ---
    with st.spinner("Adjusting prediction based on news sentiment..."):
        y_future_adjusted = adjust_prediction_with_news(y_future_raw, news_analysis)
        st.success("✅ Adjusted prediction based on news sentiment.")

    # --- 6. Plot Results ---
    st.subheader("📈 Stock Price Prediction Plot")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_hist, y_hist, 'o-', label="Historical (from Image)", color='royalblue')

    # Connect historical to prediction
    last_hist_x, last_hist_y = x_hist[-1], y_hist[-1]
    first_future_x, first_future_y = x_future[0], y_future_adjusted[0]

    ax.plot([last_hist_x, first_future_x], [last_hist_y, first_future_y], '--', color='gray')
    ax.plot(x_future, y_future_adjusted, 'x--', label="Prediction (Adjusted by News)", color='darkorange',
            markersize=8)

    ax.set_title(f"{ticker} Stock Price Prediction", fontsize=16)
    ax.set_xlabel("Time Steps (Relative)", fontsize=12)
    ax.set_ylabel("Simulated Price ($)", fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    st.pyplot(fig)

    st.success("Analysis Complete!")
    st.write("---")
    st.subheader("Final Prediction Summary:")
    st.write(f"**Last historical price:** ${y_hist[-1]:.2f}")
    st.write(f"**Predicted price after 50 time steps:** ${y_future_adjusted[-1]:.2f}")
    f"**Overall News Sentiment:** {overall_news_summary}"


# === Financial Chatbot Functions ===
def get_pacific_time():
    pacific = pytz.timezone('America/Los_Angeles')
    return datetime.now(pacific).strftime("%A, %B %d, %Y %I:%M %p %Z")

def google_search_chatbot(query, api_key, cse_id, **kwargs):
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=cse_id, **kwargs).execute()
        return res.get('items', [])
    except Exception as e:
        print(f"❌ Google Search error: {e}")
        return []

def browse_page(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        texts = [p.get_text(strip=True) for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])]
        content = ' '.join(texts)
        return re.sub(r'\s+', ' ', content).strip()[:4000]
    except requests.RequestException as e:
        print(f"❌ Browsing error: {e}")
        return None

def GoogleSearchAndBrowse(query, max_results=5, target_ticker=None): # Increased max_results and added target_ticker
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    if not google_api_key or not google_cse_id:
        return "Error: Google API keys not configured for browsing."

    current_time_pacific = get_pacific_time()
    query_with_time = f"{query} as of {current_time_pacific}"
    
    # Enhance query if a target ticker is provided
    if target_ticker:
        query_with_time = f"{target_ticker} stock {query_with_time}"

    results = google_search_chatbot(query_with_time, google_api_key, google_cse_id, num=max_results)
    
    if not results:
        return f"🕸️ No Google search results found for query: '{query_with_time}'."
    
    # Iterate through results and try to browse until relevant content is found
    for i, result in enumerate(results):
        url = result.get('link')
        if url:
            content = browse_page(url)
            if content:
                # Basic relevance check: does the content mention the target ticker if one was provided?
                # Also check for common stock finance keywords to ensure it's not a random page
                if target_ticker and \
                   (target_ticker.lower() not in content.lower() and f"{target_ticker} stock".lower() not in content.lower()) or \
                   not any(keyword in content.lower() for keyword in ["stock", "market", "finance", "price", "earnings", "investing"]):
                    logging.info(f"Skipping irrelevant search result {url} (missing ticker or finance keywords: {target_ticker})")
                    continue # Skip to the next URL if ticker or finance keywords not found in content

                return f"Search Result for '{query_with_time}' (Source {i+1}/{len(results)}):\nSource: {url}\nContent: {content}"
        
    return f"Could not retrieve relevant content from any of the top {max_results} search results for query: '{query_with_time}'."


def getRealtimeStockData(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        data = stock.info
        if not data:
            return f"No data found for ticker '{ticker}'."
        price = data.get('regularMarketPrice')
        open_price = data.get('regularMarketOpen')
        day_high = data.get('regularMarketDayHigh')
        day_low = data.get('regularMarketLow')
        volume = data.get('regularMarketVolume')
        market_time_ts = data.get('regularMarketTime')
        if price is None:
            return f"Price data not available for '{ticker}'."
        if market_time_ts:
            dt = datetime.fromtimestamp(market_time_ts, pytz.utc).astimezone(pytz.timezone('America/New_York'))
            time_str = dt.strftime("%Y-%m-%d %I:%M %p %Z")
        else:
            time_str = "N/A"
        return (
            f"**Real-time data for {ticker.upper()}:**\n\n"
            f"**Current Price:** ${price:.2f}\n"
            f"**Today's Range:** ${day_low:.2f} - ${day_high:.2f}\n"
            f"**Opening Price:** ${open_price:.2f}\n"
            f"**Trading Volume:** {volume:,} shares\n"
            f"**Last Updated:** {time_str}"
        )
    except Exception as e:
        return f"Error fetching real-time data for {ticker}: {e}"

def getHistoricalStockData(ticker: str, period: str = "1mo"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df.empty:
            return f"No historical data for '{ticker}' for period '{period}'.", {}
        df.reset_index(inplace=True)
        
        # Explicitly convert Date column to string to avoid NaT issues with default format
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        # Use .loc for setting values on a copy to avoid SettingWithCopyWarning
        df_to_display = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Apply formatting to numerical columns before converting to string
        for col in ['Open', 'High', 'Low', 'Close']:
            df_to_display.loc[:, col] = df_to_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        df_to_display.loc[:, 'Volume'] = df_to_display['Volume'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")

        # Now, convert to string
        output_string = f"Historical data for {ticker.upper()} ({period}):\n{df_to_display.to_string(index=False)}"
        
        # Calculate summary statistics for the AI to use
        summary_stats = {}
        if not df.empty:
            # Ensure numerical columns are truly numerical for calculations
            # Use original df for calculations, not df_to_display which has string values
            summary_stats['last_close'] = df['Close'].iloc[-1] if 'Close' in df.columns and not df['Close'].empty else None
            summary_stats['period_high'] = df['High'].max() if 'High' in df.columns and not df['High'].empty else None
            summary_stats['period_low'] = df['Low'].min() if 'Low' in df.columns and not df['Low'].empty else None
            summary_stats['period_volume'] = df['Volume'].sum() if 'Volume' in df.columns and not df['Volume'].empty else None
            
            # Calculate overall change for the period
            if len(df) > 1 and 'Open' in df.columns and 'Close' in df.columns:
                start_price = df['Open'].iloc[0]
                end_price = df['Close'].iloc[-1]
                if pd.notna(start_price) and pd.notna(end_price) and start_price != 0:
                    change = end_price - start_price
                    percent_change = (change / start_price) * 100
                    summary_stats['overall_change'] = f"{change:.2f}"
                    summary_stats['overall_percent_change'] = f"{percent_change:.2f}%"
                else:
                    summary_stats['overall_change'] = "N/A"
                    summary_stats['overall_percent_change'] = "N/A"
            else:
                summary_stats['overall_change'] = "N/A"
                summary_stats['overall_percent_change'] = "N/A"

        return output_string, summary_stats
    except Exception as e:
        # Log the full traceback for debugging
        logging.error(f"Error fetching historical data for {ticker} ({period}): {e}", exc_info=True)
        return f"Error fetching historical data for {ticker} ({period}): {e}", {}

def calculateInvestmentGainLoss(ticker: str, amount_usd: float, months_ago: int = 1):
    end_date = datetime.today()
    start_date = end_date - relativedelta(months=months_ago)
    try:
        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        if df.empty or len(df) < 2:
            return f"Not enough data for {ticker.upper()} to calculate investment returns."
        start_price = df.iloc[0]['Close']
        end_price = df.iloc[-1]['Close']
        shares = amount_usd / start_price
        current_value = shares * end_price
        profit = current_value - amount_usd
        percent = (profit / amount_usd) * 100
        return (
            f"Investment summary for {ticker.upper()} from {start_date.date()} to {end_date.date()}:\n"
            f"- Buy price: ${start_price:.2f}\n"
            f"- Current price: ${end_price:.2f}\n"
            f"- Shares purchased: {shares:.2f}\n"
            f"- Current value: ${current_value:.2f}\n"
            f"- {'Gain' if profit > 0 else 'Loss'}: ${abs(profit):.2f} ({percent:.2f}%)"
        )
    except Exception as e:
        return f"Error calculating investment gain/loss for {ticker}: {e}"

# === Tools Definition for Chatbot ===
# Removed GoogleSearchAndBrowse from here, as it will be called directly.
tools = [
    {
        "type": "function",
        "function": {
            "name": "getRealtimeStockData",
            "description": "Get real-time stock data for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. AAPL."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getHistoricalStockData",
            "description": "Get historical stock data for a ticker symbol over a period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    },
                    "period": {
                        "type": "string",
                        "description": "Data period, e.g. '1mo', '3mo', '1y'.",
                        "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculateInvestmentGainLoss",
            "description": "Calculate investment gain or loss for a stock over a specified number of months.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    },
                    "amount_usd": {
                        "type": "number",
                        "description": "Amount invested in USD."
                    },
                    "months_ago": {
                        "type": "integer",
                        "description": "Number of months ago the investment was made (default 1)."
                    }
                },
                "required": ["ticker", "amount_usd"]
            }
        }
    }
]

# === Chatbot Core Logic ===
def run_conversation(current_chat_history): # current_chat_history is st.session_state.messages
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")
    if not deepseek_api_key or not deepseek_base_url:
        st.error("DeepSeek API keys not configured for chatbot.")
        st.markdown("""
        **Required Environment Variables for Chatbot:**
        - `GOOGLE_API_KEY`
        - `GOOGLE_CSE_ID`
        - `DEEPSEEK_API_KEY`
        - `DEEPSEEK_BASE_URL` (e.g., `https://api.deepseek.com`)
        """)
        return {"role": "assistant", "content": "I cannot function without DeepSeek API keys. Please set them in your environment."}

    client = openai.OpenAI(api_key=deepseek_api_key, base_url=deepseek_base_url)

    # Prepare messages for the OpenAI API call, converting from simple dicts to OpenAI's expected format
    api_messages = []
    for msg in current_chat_history:
        # DeepSeek does not use OpenAI's tool_calls structure directly.
        # We only send user and assistant messages for text generation.
        # Tool outputs are pre-processed and added as part of the assistant's content.
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=api_messages, 
            temperature=0.7,
            timeout=120.0
        )
        response_message = response.choices[0].message

        final_assistant_message = {"role": response_message.role, "content": response_message.content if response_message.content is not None else ""}
        return final_assistant_message
    except openai.APITimeoutError:
        return {"role": "assistant", "content": "The AI request timed out. Please try again."}
    except Exception as e:
        return {"role": "assistant", "content": f"An error occurred: {e}"}


# === Streamlit App Functions ===
def predictor_app():
    st.title("AI-Powered Stock Price Predictor")
    st.markdown("""
    Upload a stock chart image and enter a ticker symbol to get a future price prediction,
    adjusted by real-time news sentiment analysis.
    """)

    st.sidebar.header("Inputs")
    uploaded_file = st.sidebar.file_uploader("Upload a Stock Chart Image", type=["png", "jpg", "jpeg"])
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL, GOOGL, TSLA)", "AAPL")

    if st.sidebar.button("Run Analysis"):
        if uploaded_file is not None and ticker:
            run_analysis_streamlit(uploaded_file, ticker)
        else:
            st.sidebar.warning("Please upload an image and enter a ticker symbol.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### About")
    st.sidebar.info(
        "This application uses Computer Vision (OpenCV, Tesseract OCR) to extract historical stock data from an image, "
        "linear regression for initial prediction, and AI (DeepSeek API) for real-time news sentiment analysis "
        "to adjust the future price forecast."
    )


def chatbot_app():
    st.title("Financial Chatbot")
    st.markdown("Ask me anything about stocks or general financial news!")
    st.markdown(f"Current Pacific Time: {get_pacific_time()}")

    # Initialize chat history and flags
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "waiting_for_ticker" not in st.session_state:
        st.session_state.waiting_for_ticker = False
    if "last_requested_period" not in st.session_state:
        st.session_state.last_requested_period = "1mo" # Default

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What's on your mind?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt) # Display immediately

        # Flag to check if a tool was executed
        tool_executed = False
        tool_output = None
        tool_summary_for_ai = "" # New variable to hold summary for AI

        # --- 0. Generic Ticker Extraction ---
        # Try to find a ticker symbol (2-5 uppercase letters) anywhere in the prompt
        ticker_match = re.search(r'\b([A-Z]{2,5})\b', prompt)
        extracted_ticker = ticker_match.group(1).upper() if ticker_match else None

        # --- Check for follow-up ticker for historical data ---
        # If we were waiting for a ticker AND the current prompt is just a ticker
        if st.session_state.waiting_for_ticker and extracted_ticker and len(prompt.strip()) == len(extracted_ticker):
            st.chat_message("assistant").write(f"Fetching historical data for {extracted_ticker} over {st.session_state.last_requested_period}...")
            raw_tool_output, summary_stats = getHistoricalStockData(extracted_ticker, st.session_state.last_requested_period)
            tool_output = raw_tool_output
            if summary_stats:
                tool_summary_for_ai = (
                    f"The user has been provided with historical stock data for {extracted_ticker} for the period {st.session_state.last_requested_period}.\n"
                    f"Key facts from this data:\n"
                    f"- Last closing price: ${summary_stats.get('last_close', 'N/A'):.2f}\n"
                    f"- Highest price in this period: ${summary_stats.get('period_high', 'N/A'):.2f}\n"
                    f"- Lowest price in this period: ${summary_stats.get('period_low', 'N/A'):.2f}\n"
                    f"- Total trading volume in this period: {summary_stats.get('period_volume', 'N/A'):,.0f} shares\n"
                    f"- Overall price change for the period: {summary_stats.get('overall_change', 'N/A')} ({summary_stats.get('overall_percent_change', 'N/A')})\n"
                    f"Please provide a concise, conversational summary or analysis of this data. Do NOT regenerate the table or invent any numbers. Refer ONLY to the facts provided or general market trends relevant to the provided data."
                )
            tool_executed = True
            st.session_state.waiting_for_ticker = False # Reset flag

        # --- If not a follow-up, proceed with normal intent recognition ---
        if not tool_executed:
            # 1. Real-time stock data
            if extracted_ticker and re.search(r'\b(price|current|now|real-time)\b', prompt, re.IGNORECASE):
                st.chat_message("assistant").write(f"Fetching real-time data for {extracted_ticker}...")
                tool_output = getRealtimeStockData(extracted_ticker)
                tool_executed = True
            
            # 2. Historical stock data (Initial request)
            historical_keywords_pattern = re.compile(r'\b(historical|past|last week\'s|last\s+(\d+)\s*(day|week|month|year)s?|data|price)\b', re.IGNORECASE)
            
            if not tool_executed and historical_keywords_pattern.search(prompt):
                if extracted_ticker:
                    ticker = extracted_ticker # Use the already extracted ticker

                    period = "1mo" # Default period if no specific duration is found

                    # More flexible period extraction
                    period_match_specific = re.search(r'last\s+(\d+)\s*(day|week|month|year)s?', prompt, re.IGNORECASE)
                    if "last week's" in prompt.lower() or "last week" in prompt.lower():
                        period = "5d" # 5 trading days for "last week"
                    elif period_match_specific:
                        num = int(period_match_specific.group(1))
                        unit = period_match_specific.group(2).lower()
                        if unit == 'day':
                            period = f"{num}d"
                        elif unit == 'week':
                            calculated_days = num * 5
                            if calculated_days <= 60: 
                                period = f"{calculated_days}d"
                            else:
                                period = "3mo" # Fallback for longer week periods
                        elif unit == 'month':
                            period = f"{num}mo"
                        elif unit == 'year':
                            period = f"{num}y"
                    
                    # Ensure period is one of the valid yfinance periods if it's not already
                    valid_yfinance_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
                    if period not in valid_yfinance_periods:
                        if period.endswith('d') and int(period[:-1]) > 5:
                            period = "1mo" 
                        elif period.endswith('w'): 
                            period = "1mo"
                        elif period.endswith('mo') and int(period[:-2]) > 10:
                            period = "1y" 
                        elif period.endswith('y') and int(period[:-1]) > 10:
                            period = "max" 
                        if period not in valid_yfinance_periods:
                            period = "1mo" 

                    st.chat_message("assistant").write(f"Fetching historical data for {ticker} over {period}...")
                    raw_tool_output, summary_stats = getHistoricalStockData(ticker, period)
                    tool_output = raw_tool_output
                    if summary_stats:
                        tool_summary_for_ai = (
                            f"The user has been provided with historical stock data for {ticker} for the period {period}.\n"
                            f"Key facts from this data:\n"
                            f"- Last closing price: ${summary_stats.get('last_close', 'N/A'):.2f}\n"
                            f"- Highest price in this period: ${summary_stats.get('period_high', 'N/A'):.2f}\n"
                            f"- Lowest price in this period: ${summary_stats.get('period_low', 'N/A'):.2f}\n"
                            f"- Total trading volume in this period: {summary_stats.get('period_volume', 'N/A'):,.0f} shares\n"
                            f"- Overall price change for the period: {summary_stats.get('overall_change', 'N/A')} ({summary_stats.get('overall_percent_change', 'N/A')})\n"
                            f"Please provide a concise, conversational summary or analysis of this data. Do NOT regenerate the table or invent any numbers. Refer ONLY to the facts provided or general market trends relevant to the provided data."
                        )
                    tool_executed = True
                else:
                    # If historical data is requested but no ticker is found
                    tool_output = "Please specify a stock ticker symbol (e.g., AAPL, NVDA) for which you want historical data."
                    st.session_state.waiting_for_ticker = True # Set flag to wait for ticker
                    # Store the period if identified, so we can use it on follow-up
                    period_match_specific = re.search(r'last\s+(\d+)\s*(day|week|month|year)s?', prompt, re.IGNORECASE)
                    if "last week's" in prompt.lower() or "last week" in prompt.lower():
                        st.session_state.last_requested_period = "5d"
                    elif period_match_specific:
                        num = int(period_match_specific.group(1))
                        unit = period_match_specific.group(2).lower()
                        if unit == 'day': st.session_state.last_requested_period = f"{num}d"
                        elif unit == 'week': st.session_state.last_requested_period = f"{num*5}d" # Convert weeks to days for period
                        elif unit == 'month': st.session_state.last_requested_period = f"{num}mo"
                        elif unit == 'year': st.session_state.last_requested_period = f"{num}y"
                    else:
                        st.session_state.last_requested_period = "1mo" # Default if no period specified
                    tool_executed = True # Mark as executed to prevent fallback to general search


            # 3. Calculate investment gain/loss
            # Check for investment keywords AND if a ticker was extracted
            elif not tool_executed and extracted_ticker and re.search(r'\b(invested|gain|loss|profit)\b.*\b(\d+)\b', prompt, re.IGNORECASE):
                amount_match = re.search(r'\b(\d+)\b', prompt)
                if amount_match:
                    amount = float(amount_match.group(1))
                    months_ago_match = re.search(r'(\d+)\s*(month|year)s?\s*ago', prompt, re.IGNORECASE)
                    months_ago = 1
                    if months_ago_match:
                        num = int(months_ago_match.group(1))
                        unit = months_ago_match.group(2).lower()
                        if unit == 'year':
                            months_ago = num * 12
                        else:
                            months_ago = num
                    st.chat_message("assistant").write(f"Calculating investment gain/loss for ${amount} in {extracted_ticker} {months_ago} months ago...")
                    tool_output = calculateInvestmentGainLoss(extracted_ticker, amount, months_ago)
                    tool_executed = True
                else: # If amount not found but investment keywords are
                    tool_output = "Please specify the amount invested (e.g., 'invested $1000 in AAPL')."
                    tool_executed = True
            elif not tool_executed and re.search(r'\b(invested|gain|loss|profit)\b.*\b(\d+)\b', prompt, re.IGNORECASE) and not extracted_ticker:
                tool_output = "Please specify a stock ticker symbol (e.g., AAPL, NVDA) for which you want to calculate investment gain/loss."
                tool_executed = True

            # 4. General web search (if no other tool matches)
            if not tool_executed: # This ensures it's a fallback
                st.chat_message("assistant").write(f"Searching the web for: '{prompt}'...")
                # Pass extracted_ticker to GoogleSearchAndBrowse for better relevance
                tool_output = GoogleSearchAndBrowse(prompt, max_results=5, target_ticker=extracted_ticker) 
                tool_executed = True

        if tool_executed:
            # Append tool output to history and display it
            st.session_state.messages.append({"role": "assistant", "content": tool_output})
            with st.chat_message("assistant"):
                st.markdown(tool_output)
            
            # Now, send the full history (including user prompt and tool output) to DeepSeek
            # for a conversational response based on the tool's output.
            with st.spinner("Thinking..."):
                # Create a temporary message list for DeepSeek to guide its response
                temp_messages_for_ai = list(st.session_state.messages) # Copy current history
                
                # Add a specific instruction to the AI to prevent hallucination of tables
                # Use tool_summary_for_ai if available, otherwise a generic instruction
                if tool_summary_for_ai:
                    temp_messages_for_ai.append({
                        "role": "user",
                        "content": tool_summary_for_ai
                    })
                else:
                    temp_messages_for_ai.append({
                        "role": "user",
                        "content": "The previous assistant message contains the direct data requested. Please provide a brief, conversational summary or analysis of this data. Do NOT regenerate the table or list the raw data again. Focus on trends, significant changes, or context."
                    })

                assistant_response_dict = run_conversation(temp_messages_for_ai)
                st.session_state.messages.append(assistant_response_dict) # Append the actual AI response to the main history
                with st.chat_message(assistant_response_dict["role"]):
                    st.markdown(assistant_response_dict["content"])
        # No else block here, as tool_executed is always True due to the final fallback search
        # This ensures a response is always generated.

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### About")
    st.sidebar.info(
        "This chatbot can fetch real-time and historical stock data, calculate investment returns, "
        "and browse the web for general financial information using DeepSeek AI and Google Search."
    )


# --- Main Streamlit App Layout ---
st.set_page_config(page_title="Financial AI Suite", layout="wide")

st.sidebar.title("Choose Your Mode")
selected_mode = st.sidebar.radio(
    "Select an application:",
    ("Stock Price Predictor", "Financial Chatbot")
)

if selected_mode == "Stock Price Predictor":
    predictor_app()
elif selected_mode == "Financial Chatbot":
    chatbot_app()
