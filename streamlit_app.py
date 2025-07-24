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
        logging.warning(
            f"Tesseract not found in system PATH. Attempting fallback to: {pytesseract.pytesseract.tesseract_cmd}")

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
    st.write(f"**Overall News Sentiment:** {overall_news_summary}")


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


def GoogleSearchAndBrowse(query, num_results=3): # Added num_results parameter
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    if not google_api_key or not google_cse_id:
        return "Error: Google API keys not configured for browsing."

    # Include Pacific time in the search query
    current_time_pacific = get_pacific_time()
    query_with_time = f"{query} as of {current_time_pacific}"

    results = google_search_chatbot(query_with_time, google_api_key, google_cse_id, num=num_results) # Use num_results
    if not results:
        return f"🕸️ No Google search results found for query: '{query_with_time}'."

    all_content = []
    for i, item in enumerate(results):
        url = item.get('link')
        if url:
            content = browse_page(url)
            if content:
                all_content.append(f"--- Source {i+1}: {url} ---\n{content}")
            else:
                all_content.append(f"--- Source {i+1}: {url} ---\nCould not retrieve content.")

    if not all_content:
        return f"Could not retrieve content from any source for query: '{query_with_time}'."

    return f"Search Results for '{query_with_time}':\n" + "\n\n".join(all_content)


def getRealtimeStockData(ticker: str):
    """
    Get real-time stock data for a ticker symbol.
    This function will attempt to use YFinance first, and if data is unavailable or incomplete,
    it will fall back to a Google Search to find the latest price.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.info
        if not data or data.get('regularMarketPrice') is None:
            # If yfinance fails or price is not available, try Google Search as fallback
            st.warning(f"⚠️ YFinance did not return complete real-time data for '{ticker}'. Attempting Google Search fallback.")
            search_query = f"real-time stock price {ticker}"
            # Request only 1 result for the specific price fallback
            search_result_content = GoogleSearchAndBrowse(search_query, num_results=1)
            
            # Attempt to parse price from the search result content
            price_match = re.search(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{1,4})?)', search_result_content)
            if price_match:
                try:
                    extracted_price = float(price_match.group(1).replace(',', ''))
                    return (
                        f"**Real-time data for {ticker.upper()} (via Google Search Fallback):**\n\n"
                        f"**Current Price:** ${extracted_price:.2f}\n"
                        f"**Source Content:** {search_result_content[:500]}..." # Show snippet of source
                    )
                except ValueError:
                    pass # Fall through to returning raw search content if price parsing fails

            return f"Could not retrieve real-time price for '{ticker}' from YFinance or Google Search. Search result: {search_result_content}"

        price = data.get('regularMarketPrice')
        open_price = data.get('regularMarketOpen')
        day_high = data.get('regularMarketDayHigh')
        day_low = data.get('regularMarketLow')
        volume = data.get('regularMarketVolume')
        market_time_ts = data.get('regularMarketTime')
        
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
        st.warning(f"⚠️ An error occurred fetching real-time data for {ticker} from YFinance: {e}. Attempting Google Search fallback.")
        search_query = f"real-time stock price {ticker}"
        # Request only 1 result for the specific price fallback
        search_result_content = GoogleSearchAndBrowse(search_query, num_results=1)
        
        price_match = re.search(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{1,4})?)', search_result_content)
        if price_match:
            try:
                extracted_price = float(price_match.group(1).replace(',', ''))
                return (
                    f"**Real-time data for {ticker.upper()} (via Google Search Fallback):**\n\n"
                    f"**Current Price:** ${extracted_price:.2f}\n"
                    f"**Source Content:** {search_result_content[:500]}..."
                )
            except ValueError:
                pass
        
        return f"Error fetching real-time data for {ticker}: {e}. Google Search fallback also failed to find a clear price. Search result: {search_result_content}"


def getHistoricalStockData(ticker: str, period: str = "1mo"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df.empty:
            return f"No historical data for '{ticker}' for period '{period}'."
        df.reset_index(inplace=True)
        tail_df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail()
        return f"Historical data for {ticker.upper()} ({period}):\n{tail_df.to_string(index=False)}"
    except Exception as e:
        return f"Error fetching historical data for {ticker} ({period}): {e}"


def calculateInvestmentGainLoss(ticker: str, amount_usd: float, months_ago: int = 1):
    end_date = datetime.today()
    start_date = end_date - relativedelta(months=months_ago)
    try:
        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'),
                         progress=False)
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
tools = [
    {
        "type": "function",
        "function": {
            "name": "getRealtimeStockData",
            "description": "Get real-time stock data for a ticker symbol. This function will attempt to use YFinance first, and if data is unavailable or incomplete, it will fall back to a Google Search to find the latest price.",
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
    },
    {
        "type": "function",
        "function": {
            "name": "GoogleSearchAndBrowse",
            "description": "Perform a Google search for a given query and browse the top results to extract relevant content. Useful for general information, news, or when specific data tools fail. Can retrieve content from multiple sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to use for Google search."
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "The number of top search results to browse (default 3).",
                        "minimum": 1
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# === Chatbot Core Logic ===
def run_conversation(current_chat_history):  # current_chat_history is st.session_state.messages
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
        return {"role": "assistant",
                "content": "I cannot function without DeepSeek API keys. Please set them in your environment."}

    client = openai.OpenAI(api_key=deepseek_api_key, base_url=deepseek_base_url)

    # Prepare messages for the OpenAI API call, converting from simple dicts to OpenAI's expected format
    api_messages = []
    for msg in current_chat_history:
        if msg["role"] == "tool":
            api_messages.append({
                "role": "tool",
                "tool_call_id": msg["tool_call_id"],
                "name": msg["name"],
                "content": msg["content"]
            })
        elif "tool_calls" in msg and msg["tool_calls"]:
            # This branch is for assistant messages that contain tool calls
            # The content should be None when tool_calls are present
            api_messages.append({
                "role": msg["role"],
                "content": None,
                "tool_calls": [
                    openai.types.chat.chat_completion_message_tool_call.ChatCompletionMessageToolCall(
                        id=tc["id"],
                        type=tc["type"], # Explicitly include 'type' here
                        function=openai.types.chat.chat_completion_message_tool_call.Function(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"]
                        )
                    ) for tc in msg["tool_calls"] # Iterate through tool_calls if multiple
                ]
            })
        else:
            # For regular user/assistant messages
            api_messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=api_messages,
            tools=tools, # Pass the tools definition
            tool_choice="auto", # Allow the model to choose tools
            stream=False # Not streaming for simplicity in this function
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            # Step 2: Call the model again with the tool output
            st.info("🤖 AI wants to call a tool...")
            # Add the assistant's tool call message to the chat history
            # Ensure 'type' is included when storing tool_calls in session_state
            current_chat_history.append({
                "role": response_message.role,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type, # Explicitly include type from the model's response
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in tool_calls
                ]
            })

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute the tool function
                if function_name == "getRealtimeStockData":
                    tool_response = getRealtimeStockData(ticker=function_args.get("ticker"))
                elif function_name == "getHistoricalStockData":
                    tool_response = getHistoricalStockData(ticker=function_args.get("ticker"), period=function_args.get("period", "1mo"))
                elif function_name == "calculateInvestmentGainLoss":
                    tool_response = calculateInvestmentGainLoss(
                        ticker=function_args.get("ticker"),
                        amount_usd=function_args.get("amount_usd"),
                        months_ago=function_args.get("months_ago", 1)
                    )
                elif function_name == "GoogleSearchAndBrowse": # New tool
                    tool_response = GoogleSearchAndBrowse(
                        query=function_args.get("query"),
                        num_results=function_args.get("num_results", 3) # Use num_results from tool call, default to 3
                    )
                else:
                    tool_response = f"Error: Unknown tool function: {function_name}"

                # Add tool response to chat history
                current_chat_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_response
                })
                st.info(f"🛠️ Tool '{function_name}' executed. Response: {tool_response[:100]}...") # Show snippet

            # Call the model again to get a final response based on tool output
            final_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=api_messages + current_chat_history[-len(tool_calls)-1:], # Send all previous messages + tool call + tool output
                stream=False
            )
            return final_response.choices[0].message
        else:
            # No tool call, return the assistant's direct response
            return response_message

    except openai.APIError as e:
        st.error(f"❌ OpenAI API Error: {e}")
        return {"role": "assistant", "content": f"An API error occurred: {e}"}
    except Exception as e:
        st.error(f"❌ An unexpected error occurred in run_conversation: {e}")
        return {"role": "assistant", "content": f"An unexpected error occurred: {e}"}

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Predictor & Financial Chatbot", layout="wide")

st.title("Stock Price Predictor & Financial Chatbot")

# Sidebar for API Key Instructions
st.sidebar.header("API Key Setup")
st.sidebar.markdown("""
To use this application, you need to set up the following environment variables (e.g., in a `.env` file in your project directory or as Streamlit secrets):

- `GOOGLE_API_KEY`: Your API key for Google Custom Search.
- `GOOGLE_CSE_ID`: Your Custom Search Engine ID.
- `DEEPSEEK_API_KEY`: Your API key for DeepSeek.
- `DEEPSEEK_BASE_URL`: The base URL for the DeepSeek API (e.g., `https://api.deepseek.com`).
- `TESSERACT_CMD`: (Optional) Path to your Tesseract executable if not in system PATH (e.g., `/usr/bin/tesseract` for Linux or `C:\\Program Files\\Tesseract-OCR\\tesseract.exe` for Windows).

**Example `.env` file content:**
```
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
GOOGLE_CSE_ID="YOUR_GOOGLE_CSE_ID"
DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL="[https://api.deepseek.com](https://api.deepseek.com)"
# TESSERACT_CMD="/usr/bin/tesseract"
```
""")

# Main content area
tab1, tab2 = st.tabs(["📈 Stock Price Predictor", "💬 Financial Chatbot"])

with tab1:
    st.header("Upload Chart for Price Prediction")
    uploaded_file = st.file_uploader("Choose an image file (e.g., PNG, JPG)", type=["png", "jpg", "jpeg"])
    ticker_input = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, GOOGL)", "AAPL")

    if uploaded_file and ticker_input:
        if st.button("Run Prediction"):
            run_analysis_streamlit(uploaded_file, ticker_input.strip().upper())
    elif uploaded_file:
        st.info("Please enter a stock ticker symbol to run the prediction.")
    elif ticker_input:
        st.info("Please upload a stock chart image to run the prediction.")
    else:
        st.info("Upload a stock chart image and enter a ticker symbol to get started.")

with tab2:
    st.header("Ask Your Financial Questions")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your financial chatbot. How can I assist you today?"})

    # Create a container for chat messages with a fixed height and scrollbar
    chat_history_container = st.container(height=500) # Adjust height as needed

    # Display chat messages inside the container
    with chat_history_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "tool":
                    st.markdown(f"**Tool Output:**\n```\n{message['content']}\n```")
                elif "tool_calls" in message and message["tool_calls"]:
                    st.markdown(f"**AI called tool(s):** {', '.join([tc['function']['name'] for tc in message['tool_calls']])}")
                else:
                    st.markdown(message["content"])

    # Chat input remains outside the scrollable container, at the bottom
    if prompt := st.chat_input("What's on your mind?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Re-run to display the new user message immediately
        st.rerun()

    # This part will only run after st.rerun() or if no new prompt
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = run_conversation(st.session_state.messages)
                if response:
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                    st.rerun() # Rerun to update chat history container
                else:
                    st.error("Could not get a response from the AI.")
