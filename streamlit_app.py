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
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        logging.info(f"Pytesseract command set to: {tesseract_path}")
    else:
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
        logging.warning(
            f"Tesseract not found in system PATH. Attempting fallback to: {pytesseract.pytesseract.tesseract_cmd}")

    if not pytesseract.pytesseract.tesseract_cmd:
        raise FileNotFoundError("Tesseract executable path not found or set.")

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


def extract_price_curve(image):
    if image is None:
        raise ValueError("Image object is None.")

    image_np = np.array(image)
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif image_np.ndim == 2:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)
    else:
        image_cv = image_np

    h, w, _ = image_cv.shape

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
        min_price_val = 2.6
        max_price_val = 3.8
        min_price_pixel = h * 0.9
        max_price_pixel = h * 0.1
        logging.warning(f"Falling back to simulated range: {min_price_val}-{max_price_val} due to OCR failure.")
    else:
        y_axis_labels.sort(key=lambda x: x['y_pixel'])
        min_price_val = y_axis_labels[-1]['value']
        min_price_pixel = y_axis_labels[-1]['y_pixel']
        max_price_val = y_axis_labels[0]['value']
        max_price_pixel = y_axis_labels[0]['y_pixel']

        if min_price_pixel == max_price_pixel or min_price_val == max_price_val:
            logging.warning("Insufficient distinct Y-axis labels for precise scaling. Falling back to simulated range.")
            min_price_val = 2.6
            max_price_val = 3.8
            min_price_pixel = h * 0.9
            max_price_pixel = h * 0.1

    logging.info(
        f"OCR detected Y-axis ({selected_axis_side}): Min Price: {min_price_val} at Y-pixel {min_price_pixel}, Max Price: {max_price_val} at Y-pixel {max_price_pixel}")

    chart_roi_x_start = int(w * 0.15)
    chart_roi_x_end = int(w * 0.85)
    chart_roi_y_start = int(h * 0.4)
    chart_roi_y_end = int(h * 0.85)

    chart_area = image_cv[chart_roi_y_start:chart_roi_y_end, chart_roi_x_start:chart_roi_x_end]

    if chart_area.size == 0:
        logging.error("Chart ROI is empty. Check image dimensions or ROI coordinates.")
        raise ValueError("Chart ROI is empty, cannot perform line detection.")

    hsv = cv2.cvtColor(chart_area, cv2.COLOR_BGR2HSV)

    lower_line_color1 = np.array([0, 150, 100])
    upper_line_color1 = np.array([10, 255, 255])

    lower_line_color2 = np.array([170, 150, 100])
    upper_line_color2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_line_color1, upper_line_color1)
    mask2 = cv2.inRange(hsv, lower_line_color2, upper_line_color2)
    line_mask = mask1 + mask2

    # Define kernel here, before its first use
    kernel = np.ones((2, 2), np.uint8)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logging.warning("No distinct colored line contours found. Falling back to general edge detection.")
        gray_chart_area = cv2.cvtColor(chart_area, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray_chart_area)
        blurred_chart_area = cv2.GaussianBlur(contrast_enhanced, (3, 3), 0)
        edges = cv2.Canny(blurred_chart_area, 20, 80)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found in the chart area after color and fallback edge detection.")

    all_line_points = []
    for contour in contours:
        if cv2.contourArea(contour) > 1:
            for pt in contour:
                all_line_points.append((pt[0][0] + chart_roi_x_start, pt[0][1] + chart_roi_y_start))

    if not all_line_points:
        raise ValueError("No valid points extracted from the main line after cleaning.")

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

    y_vals_real = np.interp(y_pixel_vals,
                            (max_price_pixel, min_price_pixel),
                            (max_price_val, min_price_val)
                            ).tolist()

    x_vals = list(range(len(x_pixel_vals)))

    return x_vals, y_vals_real


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


def run_analysis_streamlit(uploaded_file, ticker):
    st.info(f"🔎 Starting full analysis for {ticker} using the uploaded image.")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")
    alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not all([google_api_key, google_cse_id, deepseek_api_key, deepseek_base_url, alpha_vantage_api_key]):
        st.error(
            "Missing one or more required environment variables. Please check your `.env` file or Streamlit secrets.")
        st.markdown("""
        **Required Environment Variables for Predictor:**
        - `GOOGLE_API_KEY`
        - `GOOGLE_CSE_ID`
        - `DEEPSEEK_API_KEY`
        - `DEEPSEEK_BASE_URL` (e.g., `https://api.deepseek.com`)
        - `ALPHA_VANTAGE_API_KEY` (Get a free key from [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key))
        - `TESSERACT_CMD` (Optional, if tesseract is not in your system's PATH, though auto-detection is preferred)
        """)
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Stock Chart", use_column_width=True)

    with st.spinner("Extracting historical data from image..."):
        try:
            x_hist, y_hist = extract_price_curve(image)
            st.success(f"✅ Extracted {len(x_hist)} data points from the chart image.")
        except (FileNotFoundError, ValueError) as e:
            st.error(f"❌ Failed to extract data from image: {e}")
            return

    with st.spinner("Generating initial price prediction..."):
        x_future, y_future_raw = predict_future_prices(x_hist, y_hist, num_future=50)
        st.success("✅ Generated initial future price prediction.")

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

    with st.spinner("Adjusting prediction based on news sentiment..."):
        y_future_adjusted = adjust_prediction_with_news(y_future_raw, news_analysis)
        st.success("✅ Adjusted prediction based on news sentiment.")

    st.subheader("📈 Stock Price Prediction Plot")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_hist, y_hist, 'o-', label="Historical (from Image)", color='royalblue')

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


def get_pacific_time():
    pacific = pytz.timezone('America/Los_Angeles')
    return datetime.now(pacific).strftime("%A, %B %d, %Y %I:%M %p %Z")

def getCurrentPacificTime():
    return get_pacific_time()


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
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'form', 'button', 'img', 'svg', 'aside', 'noscript', 'meta', 'link', 'input', 'select', 'textarea']):
            tag.decompose()

        main_content_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article', 'section', 'main', 'div', 'span'])
        
        texts = [tag.get_text(strip=True) for tag in main_content_tags if tag.get_text(strip=True)]

        content = ' '.join(texts)
        cleaned_content = re.sub(r'\s+', ' ', content).strip()

        return cleaned_content[:4000]
    except requests.RequestException as e:
        print(f"❌ Browsing error: {e}")
        return None


def GoogleSearchAndBrowse(query, num_results=3):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    if not google_api_key or not google_cse_id:
        return "Error: Google API keys not configured for browsing."

    current_time_pacific = get_pacific_time()
    query_with_time = f"{query} as of {current_time_pacific}"

    results = google_search_chatbot(query_with_time, google_api_key, google_cse_id, num=num_results)
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

def getRealtimeStockData_AlphaVantage(ticker: str, api_key: str):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            price = float(quote.get("05. price"))
            open_price = float(quote.get("02. open"))
            high_price = float(quote.get("03. high"))
            low_price = float(quote.get("04. low"))
            volume = int(quote.get("06. volume"))
            latest_trading_day = quote.get("07. latest trading day")
            
            time_str = f"End of Day {latest_trading_day}" if latest_trading_day else "N/A"

            return {
                "price": price,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "volume": volume,
                "time_str": time_str,
                "source": "Alpha Vantage"
            }
        elif "Error Message" in data:
            logging.error(f"Alpha Vantage Error for {ticker}: {data['Error Message']}")
            return {"error": data['Error Message']}
        elif "Note" in data:
            logging.warning(f"Alpha Vantage Note for {ticker}: {data['Note']}")
            return {"error": data['Note']}
        else:
            return {"error": "No 'Global Quote' data found in Alpha Vantage response."}

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Alpha Vantage API request error for {ticker}: {e}")
        return {"error": f"Alpha Vantage API request failed: {e}"}
    except ValueError as e:
        logging.error(f"❌ Alpha Vantage data parsing error for {ticker}: {e}")
        return {"error": f"Alpha Vantage data parsing failed: {e}"}
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred with Alpha Vantage for {ticker}: {e}")
        return {"error": f"An unexpected error occurred with Alpha Vantage: {e}"}

def _scrape_price_from_url(url: str, selector: str, ticker: str, source_name: str):
    try:
        headers = {'User-Agent': 'Mozilla/50 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        price_element = soup.select_one(selector)
        
        if price_element:
            price_text = price_element.get_text(strip=True)
            cleaned_price_str = re.sub(r'[^\d.,]', '', price_text)
            if ',' in cleaned_price_str and '.' in cleaned_price_str:
                if cleaned_price_str.rfind(',') > cleaned_price_str.rfind('.'):
                    cleaned_price_str = cleaned_price_str.replace('.', '').replace(',', '.')
                else:
                    cleaned_price_str = cleaned_price_str.replace(',', '')
            
            cleaned_price_str = cleaned_price_str.replace(',', '')
            price = float(cleaned_price_str)
            return {"price": price, "source": source_name}
        return None
    except requests.RequestException as e:
        logging.warning(f"Failed to scrape {url} for {ticker}: {e}")
        return None
    except ValueError as e:
        logging.warning(f"Could not parse price from {url} for {ticker}: {e}")
        return None

def getRealtimeStockData(ticker: str):
    alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if alpha_vantage_api_key:
        data = getRealtimeStockData_AlphaVantage(ticker, alpha_vantage_api_key)
        if data and "error" not in data:
            return data
        elif data and "error" in data:
            pass
        else:
            pass

    sources = [
        {"url": f"https://finance.yahoo.com/quote/{ticker}/", "selector": f"fin-qsp-price[data-symbol='{ticker}']", "name": "Yahoo Finance"},
        {"url": f"https://www.google.com/finance/quote/{ticker}:NASDAQ", "selector": "div[data-source='inline_price']", "name": "Google Finance"},
        {"url": f"https://www.marketwatch.com/investing/stock/{ticker}", "selector": "bg-quote.value", "name": "MarketWatch"}
    ]

    for source in sources:
        price_data = _scrape_price_from_url(source["url"], source["selector"], ticker, source["name"])
        if price_data:
            return {
                "price": price_data["price"],
                "open": "N/A",
                "high": "N/A",
                "low": "N/A",
                "volume": "N/A",
                "time_str": get_pacific_time(),
                "source": price_data["source"]
            }
    return {"error": f"Could not retrieve real-time data for {ticker} from any source."}


def getHistoricalStockData(ticker: str, period: str = "1y"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return f"No historical data found for {ticker} for the period {period}."
        
        hist.index = hist.index.strftime('%Y-%m-%d')
        hist_df = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
        hist_df.columns = [f'{ticker} Open', f'{ticker} High', f'{ticker} Low', f'{ticker} Close', f'{ticker} Volume']
        
        return hist_df.to_markdown()
    except Exception as e:
        return f"Error fetching historical data for {ticker}: {e}"

def getCompanyInfo(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            return f"No company information found for {ticker}."
        
        company_summary = {
            "Symbol": info.get("symbol"),
            "Long Name": info.get("longName"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Country": info.get("country"),
            "Full Time Employees": info.get("fullTimeEmployees"),
            "Market Cap": info.get("marketCap"),
            "Current Price": info.get("currentPrice"),
            "Recommendation Key": info.get("recommendationKey"),
            "Website": info.get("website"),
            "Business Summary": info.get("longBusinessSummary", "No business summary available.")
        }
        
        if company_summary.get("Market Cap"):
            company_summary["Market Cap"] = f"${company_summary['Market Cap']:,}"
        if company_summary.get("Full Time Employees"):
            company_summary["Full Time Employees"] = f"{company_summary['Full Time Employees']:,}"
        if company_summary.get("Current Price"):
            company_summary["Current Price"] = f"${company_summary['Current Price']:.2f}"

        info_str = f"**Company Information for {company_summary.get('Long Name', ticker)} ({company_summary.get('Symbol', '')}):**\n"
        for key, value in company_summary.items():
            if key not in ["Symbol", "Long Name", "Business Summary"]:
                info_str += f"- **{key}:** {value}\n"
        info_str += f"\n**Business Summary:**\n{company_summary['Business Summary']}"
        
        return info_str
    except Exception as e:
        return f"Error fetching company information for {ticker}: {e}"

def getFinancialStatements(ticker: str, statement_type: str = "income_statement", period: str = "annual"):
    try:
        stock = yf.Ticker(ticker)
        
        if statement_type == "income_statement":
            data = stock.income_stmt if period == "annual" else stock.quarterly_income_stmt
        elif statement_type == "balance_sheet":
            data = stock.balance_sheet if period == "annual" else stock.quarterly_balance_sheet
        elif statement_type == "cash_flow":
            data = stock.cashflow if period == "annual" else stock.quarterly_cashflow
        else:
            return "Invalid statement type. Choose from 'income_statement', 'balance_sheet', 'cash_flow'."

        if data.empty:
            return f"No {period} {statement_type} data found for {ticker}."
        
        data = data.T
        data.index.name = "Date"
        data.index = data.index.strftime('%Y-%m-%d')
        
        return f"**{period.capitalize()} {statement_type.replace('_', ' ').capitalize()} for {ticker}:**\n" + data.to_markdown()
    except Exception as e:
        return f"Error fetching financial statements for {ticker}: {e}"

def getMajorHolders(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        major_holders = stock.major_holders
        
        if major_holders.empty:
            return f"No major holders data found for {ticker}."
        
        major_holders.columns = ['Percentage', 'Description']
        major_holders_str = f"**Major Holders for {ticker}:**\n" + major_holders.to_markdown(index=False)
        return major_holders_str
    except Exception as e:
        return f"Error fetching major holders for {ticker}: {e}"

def getInstitutionalHolders(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        institutional_holders = stock.institutional_holders
        
        if institutional_holders.empty:
            return f"No institutional holders data found for {ticker}."
        
        institutional_holders.columns = ['Holder', 'Shares', 'Date Reported', 'Percentage Out', 'Value']
        institutional_holders_str = f"**Institutional Holders for {ticker}:**\n" + institutional_holders.to_markdown(index=False)
        return institutional_holders_str
    except Exception as e:
        return f"Error fetching institutional holders for {ticker}: {e}"

def getRecommendations(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        
        if recommendations.empty:
            return f"No analyst recommendations found for {ticker}."
        
        recommendations_str = f"**Analyst Recommendations for {ticker}:**\n" + recommendations.head(10).to_markdown()
        return recommendations_str
    except Exception as e:
        return f"Error fetching analyst recommendations for {ticker}: {e}"

def getDividends(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        
        if dividends.empty:
            return f"No dividend history found for {ticker}."
        
        dividends = dividends.reset_index()
        dividends.columns = ['Date', 'Dividend']
        dividends['Date'] = dividends['Date'].dt.strftime('%Y-%m-%d')
        
        dividends_str = f"**Dividend History for {ticker}:**\n" + dividends.to_markdown(index=False)
        return dividends_str
    except Exception as e:
        return f"Error fetching dividend history for {ticker}: {e}"

def getEarnings(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        
        annual_earnings = stock.earnings
        quarterly_earnings = stock.quarterly_earnings
        
        response_str = ""
        if not annual_earnings.empty:
            annual_earnings = annual_earnings.T
            annual_earnings.index.name = "Year"
            annual_earnings.index = annual_earnings.index.astype(str)
            response_str += f"**Annual Earnings for {ticker}:**\n" + annual_earnings.to_markdown() + "\n\n"
        else:
            response_str += f"No annual earnings data found for {ticker}.\n\n"
            
        if not quarterly_earnings.empty:
            quarterly_earnings = quarterly_earnings.T
            quarterly_earnings.index.name = "Quarter"
            quarterly_earnings.index = quarterly_earnings.index.strftime('%Y-%m-%d')
            response_str += f"**Quarterly Earnings for {ticker}:**\n" + quarterly_earnings.to_markdown()
        else:
            response_str += f"No quarterly earnings data found for {ticker}."
            
        return response_str
    except Exception as e:
        return f"Error fetching earnings data for {ticker}: {e}"

def getNews(ticker: str, num_articles: int = 5):
    query = f"{ticker} stock news"
    news_links = search_news(query, os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_CSE_ID"), num_results=num_articles)
    
    if not news_links:
        return f"No recent news found for {ticker}."
    
    news_str = f"**Recent News for {ticker}:**\n"
    for i, link in enumerate(news_links):
        news_str += f"{i+1}. {link}\n"
    return news_str

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")

if not deepseek_api_key or not deepseek_base_url:
    st.error("DEEPSEEK_API_KEY or DEEPSEEK_BASE_URL environment variables not set. Please set them in your `.env` file or Streamlit secrets.")
    st.stop()

client = openai.OpenAI(api_key=deepseek_api_key, base_url=deepseek_base_url)

tools = [
    {
        "type": "function",
        "function": {
            "name": "getRealtimeStockData",
            "description": "Get real-time stock data for a given stock ticker. This function attempts to fetch data from Alpha Vantage first, and falls back to web scraping if Alpha Vantage fails or is not configured. It returns the current price, open, high, low, volume, and the time of the data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)."
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
            "description": "Get historical stock data for a given stock ticker and period. Period options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)."
                    },
                    "period": {
                        "type": "string",
                        "description": "The period for historical data (e.g., '1y' for one year, '5d' for five days).",
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
            "name": "getCompanyInfo",
            "description": "Get general company information for a given stock ticker, including business summary, sector, industry, market cap, and website.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getFinancialStatements",
            "description": "Get financial statements (income statement, balance sheet, or cash flow) for a given stock ticker. Specify the statement type and period (annual or quarterly).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)."
                    },
                    "statement_type": {
                        "type": "string",
                        "description": "The type of financial statement.",
                        "enum": ["income_statement", "balance_sheet", "cash_flow"]
                    },
                    "period": {
                        "type": "string",
                        "description": "The period for the financial statement (annual or quarterly).",
                        "enum": ["annual", "quarterly"]
                    }
                },
                "required": ["ticker", "statement_type", "period"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getMajorHolders",
            "description": "Get major holders information for a given stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getInstitutionalHolders",
            "description": "Get institutional holders information for a given stock ticker, including shares held, date reported, percentage out, and value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getRecommendations",
            "description": "Get analyst recommendations for a given stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getDividends",
            "description": "Get dividend history for a given ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getEarnings",
            "description": "Get earnings history (annual and quarterly) for a given stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getNews",
            "description": "Get recent news articles for a given stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)."
                    },
                    "num_articles": {
                        "type": "integer",
                        "description": "The number of news articles to retrieve (default is 5).",
                        "default": 5
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "GoogleSearchAndBrowse",
            "description": "Perform a Google search for a given query and browse the top results to extract content. Useful for general knowledge questions or information not covered by specific stock tools. Returns a summary of search results and browsed content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "The number of top search results to browse (default is 3).",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getCurrentPacificTime",
            "description": "Get the current date and time in Pacific Time (America/Los_Angeles).",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

available_tools = {
    "getRealtimeStockData": getRealtimeStockData,
    "getHistoricalStockData": getHistoricalStockData,
    "getCompanyInfo": getCompanyInfo,
    "getFinancialStatements": getFinancialStatements,
    "getMajorHolders": getMajorHolders,
    "getInstitutionalHolders": getInstitutionalHolders,
    "getRecommendations": getRecommendations,
    "getDividends": getDividends,
    "getEarnings": getEarnings,
    "getNews": getNews,
    "GoogleSearchAndBrowse": GoogleSearchAndBrowse,
    "getCurrentPacificTime": getCurrentPacificTime
}

def run_conversation(messages):
    time.sleep(1)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message
        
        if response_message.tool_calls:
            tool_call_message_dict = {
                "role": response_message.role,
                "content": response_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "function": {
                            "arguments": tc.function.arguments,
                            "name": tc.function.name
                        },
                        "type": "function"
                    } for tc in response_message.tool_calls
                ]
            }
            messages.append(tool_call_message_dict)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_tools.get(function_name)
                
                if function_to_call:
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        tool_output = function_to_call(**function_args)
                        
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": str(tool_output),
                            }
                        )
                    except json.JSONDecodeError as e:
                        error_message = f"Error parsing tool arguments for {function_name}: {e}. Arguments: {tool_call.function.arguments}"
                        st.error(error_message)
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": f"Error: {error_message}",
                            }
                        )
                    except Exception as e:
                        error_message = f"Error executing tool {function_name}: {e}"
                        st.error(error_message)
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": f"Error: {error_message}",
                            }
                        )
                else:
                    error_message = f"Tool {function_name} not found."
                    st.error(error_message)
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: {error_message}",
                        }
                    )
            
            second_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
            )
            final_response_message_obj = second_response.choices[0].message
            return {
                "role": final_response_message_obj.role,
                "content": final_response_message_obj.content or ""
            }
        else:
            return {
                "role": response_message.role,
                "content": response_message.content or ""
            }

    except openai.APIError as e:
        st.error(f"DeepSeek API Error: {e}")
        return {"role": "assistant", "content": f"An API error occurred: {e}"}
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return {"role": "assistant", "content": f"An unexpected error occurred: {e}"}


st.set_page_config(page_title="Ask Your Financial Questions", page_icon="📈")
st.title("Financial Analysis App")

tab1, tab2 = st.tabs(["Stock Chart Image Predictor", "Financial Chatbot"])

with tab1:
    st.header("Stock Chart Image Predictor")
    st.write("Upload a stock chart image and enter a ticker to get a price prediction based on historical data and news sentiment.")
    uploaded_file = st.file_uploader("Upload a stock chart image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"], key="predictor_uploader")
    ticker_input_predictor = st.text_input("Enter stock ticker (e.g., TSLA) for image analysis:", key="ticker_predictor")
    
    if st.button("Analyze Chart"):
        if uploaded_file and ticker_input_predictor:
            run_analysis_streamlit(uploaded_file, ticker_input_predictor.upper())
        else:
            st.warning("Please upload an image and enter a stock ticker to analyze the chart.")

with tab2:
    st.header("Financial Chatbot")
    st.write("Hello! I'm your financial chatbot. How can I assist you today? You can ask me about real-time stock data, historical data, company information, financial statements, major holders, institutional holders, recommendations, dividends, earnings, or recent news for any stock.")

    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    st.markdown(
        """
        <style>
        .reportview-container .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .fixed-bottom-input {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: white;
            padding: 10px 0;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        .block-container {
            padding-bottom: 70px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown('<div class="fixed-bottom-input">', unsafe_allow_html=True)
        prompt = st.chat_input("What's on your mind?", key="chat_input")
        st.markdown('</div>', unsafe_allow_html=True)


    if prompt:
        with chat_container:
            st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            response = run_conversation(st.session_state.messages)
            
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response["content"])
            st.session_state.messages.append({"role": response["role"], "content": response["content"]})
