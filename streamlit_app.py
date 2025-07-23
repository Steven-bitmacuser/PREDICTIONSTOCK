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
import shutil # Needed for shutil.which()

# --- Basic Setup ---
# Set up logging for Streamlit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()  # Load environment variables from .env file

# --- Pytesseract Configuration with Error Handling ---
# This block ensures Tesseract OCR is correctly located, which is crucial for deployment.
try:
    # Attempt to find tesseract in the system's PATH first
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        logging.info(f"Pytesseract command set to: {tesseract_path}")
    else:
        # Fallback for environments like Streamlit Cloud where tesseract is in a fixed path
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
        logging.warning(f"Tesseract not found in system PATH. Attempting fallback to: {pytesseract.pytesseract.tesseract_cmd}")

    # Verify if tesseract_cmd is set and tesseract is accessible
    if not pytesseract.pytesseract.tesseract_cmd:
        raise FileNotFoundError("Tesseract executable path could not be determined.")
    
    # Verify Tesseract is callable by checking its version.
    pytesseract.get_tesseract_version()

except pytesseract.TesseractNotFoundError:
    st.error(
        "❌ Tesseract OCR engine not found! Please ensure Tesseract is installed and configured "
        "in your `packages.txt` for Streamlit Cloud deployment. "
        "Refer to the Streamlit documentation for deploying apps with system dependencies."
    )
    st.stop()
except FileNotFoundError as e:
    st.error(
        "❌ Pytesseract configuration error. The Tesseract executable path could not be determined. "
        "Please ensure Tesseract is installed and its path is correctly set."
    )
    st.code(f"Error details: {e}")
    st.stop()
except Exception as e:
    st.error(
        "❌ An unexpected error occurred during Pytesseract configuration. Please check your setup."
    )
    st.code(f"Error details: {e}")
    st.stop()


# --- chart_extractor.py content ---
def extract_price_curve(image):
    """
    Extracts a price curve and its real values from a chart image using computer vision and OCR.
    """
    if image is None:
        raise ValueError("Image object is None.")

    # Convert PIL Image to OpenCV format
    image_np = np.array(image)
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif image_np.ndim == 2:  # Grayscale image
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    else:
        image_cv = image_np

    h, w, _ = image_cv.shape

    # --- 1. OCR for Y-Axis Labels ---
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
        ocr_data_left = pytesseract.image_to_data(ocr_thresh_left, output_type=pytesseract.Output.DICT, config='--psm 6')
        potential_y_axes_data.append((ocr_data_left, y_axis_roi_y_start, "left"))

    right_roi = image_cv[y_axis_roi_y_start:y_axis_roi_y_end, right_y_axis_roi_x_start:right_y_axis_roi_x_end]
    if right_roi.size > 0:
        gray_right_roi = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)
        inverted_right_roi = cv2.bitwise_not(gray_right_roi)
        _, ocr_thresh_right = cv2.threshold(inverted_right_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ocr_data_right = pytesseract.image_to_data(ocr_thresh_right, output_type=pytesseract.Output.DICT, config='--psm 6')
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
        raise ValueError("No numeric Y-axis labels found via OCR on either side. Cannot determine price scale.")
    
    y_axis_labels.sort(key=lambda x: x['y_pixel'])
    min_price_val = y_axis_labels[-1]['value']
    min_price_pixel = y_axis_labels[-1]['y_pixel']
    max_price_val = y_axis_labels[0]['value']
    max_price_pixel = y_axis_labels[0]['y_pixel']

    if min_price_pixel == max_price_pixel or min_price_val == max_price_val:
        raise ValueError("Insufficient distinct Y-axis labels for precise scaling.")

    logging.info(f"OCR detected Y-axis ({selected_axis_side}): Min Price: {min_price_val}, Max Price: {max_price_val}")

    # --- 2. Extract Price Line ---
    chart_roi_x_start = int(w * 0.15)
    chart_roi_x_end = int(w * 0.85)
    chart_roi_y_start = int(h * 0.1)
    chart_roi_y_end = int(h * 0.9)
    chart_area = image_cv[chart_roi_y_start:chart_roi_y_end, chart_roi_x_start:chart_roi_x_end]

    if chart_area.size == 0:
        raise ValueError("Chart ROI is empty, cannot perform line detection.")

    hsv = cv2.cvtColor(chart_area, cv2.COLOR_BGR2HSV)
    lower_line_color1 = np.array([0, 150, 100])
    upper_line_color1 = np.array([10, 255, 255])
    lower_line_color2 = np.array([170, 150, 100])
    upper_line_color2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_line_color1, upper_line_color1)
    mask2 = cv2.inRange(hsv, lower_line_color2, upper_line_color2)
    line_mask = mask1 + mask2
    
    kernel = np.ones((2, 2), np.uint8)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logging.warning("No distinct colored line found. Falling back to general edge detection.")
        gray_chart_area = cv2.cvtColor(chart_area, cv2.COLOR_BGR2GRAY)
        blurred_chart_area = cv2.GaussianBlur(gray_chart_area, (3, 3), 0)
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
        raise ValueError("No valid points extracted from any significant contour.")

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

    # --- 3. Map Pixel Y-values to Real Price Values ---
    y_vals_real = np.interp(y_pixel_vals, (max_price_pixel, min_price_pixel), (max_price_val, min_price_val)).tolist()
    x_vals = list(range(len(x_pixel_vals)))

    return x_vals, y_vals_real


# --- prediction.py content ---
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
    if total_weight == 0:
        return y_values
    
    net_effect = 0
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    for item in news_items:
        weight = item.get("importance", 0.5)
        sentiment = item.get("sentiment", "neutral")
        impact = sentiment_map.get(sentiment.lower(), 0)
        net_effect += weight * impact

    adjustment_factor = net_effect / total_weight
    logging.info(f"🔧 News Adjustment Factor: {adjustment_factor:.2f}")
    max_adjustment = 0.02
    adjustment_percentage = max_adjustment * adjustment_factor
    return [price * (1 + adjustment_percentage) for price in y_values]


# --- news_service.py content ---
def search_news(query, api_key, cse_id, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": f"{query} stock financial news OR earnings OR analyst ratings OR market outlook", "cx": cse_id, "key": api_key, "num": num_results}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("items", [])
        return [item["link"] for item in results]
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Google Search Error: {e}")
        st.error("Google Search Error. Please check your GOOGLE_API_KEY and GOOGLE_CSE_ID.")
        st.code(f"Error details: {e}")
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
        "Return the result as a JSON object with two keys: 'overall_summary' (a concise string summarizing the collective sentiment) and 'news_items' (a list of objects, each with 'sentiment', 'importance', and 'explanation'). DO NOT include any other text or markdown formatting outside the JSON object. ONLY the JSON object.\nFor example: "
        '{"overall_summary": "Overall summary of news...", "news_items": [{"sentiment": "positive", "importance": 0.8, "explanation": "..."}, {"sentiment": "neutral", "importance": 0.6, "explanation": "..."}]}'
    )
    logging.info("Requesting news sentiment analysis from AI...")
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
    except json.JSONDecodeError as e:
        logging.error("Failed to decode JSON from AI response. Raw response:\n" + text)
        st.error("Failed to decode JSON from AI response. The response might not be valid JSON.")
        st.code(f"Error details: {e}\n\nRaw response from AI:\n{text}")
        return {"overall_summary": "Failed to parse AI response as JSON.", "news_items": []}
    except Exception as e:
        logging.error(f"An unexpected error occurred during news analysis: {e}", exc_info=True)
        st.error("An unexpected error occurred during AI analysis.")
        st.code(f"Error details: {e}")
        return {"overall_summary": f"An error occurred during AI analysis.", "news_items": []}


# --- Main Streamlit Application Logic ---
def run_analysis_streamlit(uploaded_file, ticker):
    """
    Runs the full stock analysis workflow for Streamlit.
    """
    # FIX: Add a global try-except block to catch ALL unhandled exceptions
    # during the analysis and display them safely to prevent a frontend crash.
    try:
        st.info(f"🔎 Starting full analysis for {ticker} using the uploaded image.")

        # --- 1. Load API Keys ---
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")

        if not all([google_api_key, google_cse_id, deepseek_api_key, deepseek_base_url]):
            st.error("Missing one or more required environment variables. Please check your `.env` file or Streamlit secrets.")
            return

        image = Image.open(uploaded_file)

        # --- Image Resizing for Mobile Performance ---
        MAX_IMAGE_SIZE = (1600, 1600)
        if image.width > MAX_IMAGE_SIZE[0] or image.height > MAX_IMAGE_SIZE[1]:
            original_size = image.size
            image.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
            st.info(f"🖼️ Resized image from {original_size[0]}x{original_size[1]} to {image.width}x{image.height} for performance.")

        st.image(image, caption="Uploaded Stock Chart", use_column_width=True)

        # --- 2. Extract Historical Data from Image ---
        with st.spinner("Extracting historical data from image..."):
            x_hist, y_hist = extract_price_curve(image)
            st.success(f"✅ Extracted {len(x_hist)} data points from the chart image.")

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
                    st.markdown(f"- [{link}]({link})")

                news_analysis_result = analyze_news(news_links, deepseek_api_key, deepseek_base_url)
                news_items_list = news_analysis_result.get('news_items', [])
                overall_news_summary = news_analysis_result.get('overall_summary', 'No overall summary provided.')

                if news_items_list:
                    st.success("✅ News analysis complete.")
                    st.subheader("Overall News Summary:")
                    st.text(overall_news_summary)
                    
                    st.subheader("Individual News Analysis Details:")
                    for i, item in enumerate(news_items_list):
                        st.markdown(f"**News Item {i + 1}:**")
                        st.write(f"  **Sentiment:** {item.get('sentiment', 'N/A')}")
                        st.write(f"  **Importance:** {item.get('importance', 'N/A'):.2f}")
                        st.write("  **Explanation:**")
                        st.text(item.get('explanation', 'No explanation provided.'))
                    news_analysis = news_items_list
                else:
                    st.warning("News analysis returned no results. Prediction will not be adjusted by news.")
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
        
        last_hist_x, last_hist_y = x_hist[-1], y_hist[-1]
        first_future_x, first_future_y = x_future[0], y_future_adjusted[0]
        ax.plot([last_hist_x, first_future_x], [last_hist_y, first_future_y], '--', color='gray')
        ax.plot(x_future, y_future_adjusted, 'x--', label="Prediction (Adjusted by News)", color='darkorange', markersize=8)

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
        st.write("**Overall News Sentiment:**")
        st.text(overall_news_summary)

    except Exception as e:
        # This global exception handler is the final defense against frontend crashes.
        # It catches any error from the entire workflow and displays it safely.
        logging.error(f"A critical error occurred during the analysis workflow: {e}", exc_info=True)
        st.error("An unexpected error occurred during the analysis.")
        st.info("This can sometimes happen due to issues with image parsing, API responses, or other transient problems. Please try again with a different image or ticker.")
        st.code(f"Error details: {e}")


# --- Streamlit App UI Layout ---
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

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
    "linear regression for initial prediction, and an AI API for real-time news sentiment analysis "
    "to adjust the future price forecast."
)

st.sidebar.markdown("#### API Key Setup")
st.sidebar.markdown(
    "Ensure you have a `.env` file in the same directory as this script with the following variables set, or configure them as Streamlit secrets:"
)
st.sidebar.code("""
GOOGLE_API_KEY="your_google_api_key"
GOOGLE_CSE_ID="your_google_custom_search_engine_id"
DEEPSEEK_API_KEY="your_deepseek_api_key"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
# TESSERACT_CMD is usually not needed if tesseract is in packages.txt
# TESSERACT_CMD="/usr/bin/tesseract"
""")
