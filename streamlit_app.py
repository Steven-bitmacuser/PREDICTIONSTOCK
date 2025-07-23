import os
import argparse
import logging
import json
import re
import cv2
import numpy as np
import requests
import openai
import matplotlib
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PIL import Image
import pytesseract

# Set the Matplotlib backend explicitly to avoid IDE-specific issues
matplotlib.use('TkAgg')

# Basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()  # Load environment variables from .env file

# Configure pytesseract (if Tesseract is not in your PATH)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


# --- chart_extractor.py content (Enhanced for Maximum Data Points) ---
def extract_price_curve(image_path):
    """
    Extracts a price curve and its real values from a chart image using computer vision and OCR.

    This version is highly optimized for extracting a large number of data points
    from a distinctively colored stock price line (like red/pink in WechatIMG204 2.jpg).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image from path: {image_path}")

    h, w, _ = image.shape

    # --- 1. OCR for Y-Axis Labels (Universal Detection - unchanged) ---
    left_y_axis_roi_x_start = 0
    left_y_axis_roi_x_end = int(w * 0.25)
    right_y_axis_roi_x_start = int(w * 0.75)
    right_y_axis_roi_x_end = w
    y_axis_roi_y_start = int(h * 0.1)
    y_axis_roi_y_end = int(h * 0.95)

    potential_y_axes_data = []

    left_roi = image[y_axis_roi_y_start:y_axis_roi_y_end, left_y_axis_roi_x_start:left_y_axis_roi_x_end]
    if left_roi.size > 0:
        gray_left_roi = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
        inverted_left_roi = cv2.bitwise_not(gray_left_roi)
        _, ocr_thresh_left = cv2.threshold(inverted_left_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ocr_data_left = pytesseract.image_to_data(ocr_thresh_left, output_type=pytesseract.Output.DICT,
                                                  config='--psm 6')
        potential_y_axes_data.append((ocr_data_left, y_axis_roi_y_start, "left"))

    right_roi = image[y_axis_roi_y_start:y_axis_roi_y_end, right_y_axis_roi_x_start:right_y_axis_roi_x_end]
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

    chart_area = image[chart_roi_y_start:chart_roi_y_end, chart_roi_x_start:chart_roi_x_end]

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
        return {"overall_summary": "Failed to parse AI response as JSON.", "news_items": []}
    except openai.APITimeoutError:
        logging.error(
            "DeepSeek API request timed out (exceeded 120 seconds). Please check your internet connection or try again later.")
        return {"overall_summary": "AI request timed out.", "news_items": []}
    except Exception as e:
        logging.error(f"An unexpected error occurred during news analysis: {e}")
        return {"overall_summary": f"An error occurred during AI analysis: {e}", "news_items": []}


# --- main.py content (adapted) ---
def run_analysis(image_path, ticker):
    """
    Runs the full stock analysis workflow.
    """
    logging.info(f"🔎 Starting full analysis for {ticker} using '{image_path}'")

    # --- 1. Load API Keys ---
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")

    if not all([google_api_key, google_cse_id, deepseek_api_key, deepseek_base_url]):
        logging.error("Missing one or more required environment variables. Check your .env file.")
        logging.error("Please ensure GOOGLE_API_KEY, GOOGLE_CSE_ID, DEEPSEEK_API_KEY, and DEEPSEEK_BASE_URL are set.")
        return

    # --- 2. Extract Historical Data from Image ---
    try:
        x_hist, y_hist = extract_price_curve(image_path)
        logging.info(f"✅ Extracted {len(x_hist)} data points from the chart image.")
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"❌ Failed to extract data from image: {e}")
        return

    # --- 3. Make Initial Price Prediction ---
    x_future, y_future_raw = predict_future_prices(x_hist, y_hist, num_future=50)
    logging.info("✅ Generated initial future price prediction.")

    # --- 4. Get and Analyze News ---
    news_links = search_news(f"{ticker} stock news", google_api_key,
                             google_cse_id)
    if news_links:
        logging.info(f"📰 Found {len(news_links)} top news links:\n" + "\n".join(f"- {link}" for link in news_links))
        news_analysis_result = analyze_news(news_links, deepseek_api_key, deepseek_base_url)

        news_items_list = news_analysis_result.get('news_items', [])
        overall_summary = news_analysis_result.get('overall_summary', 'No overall summary provided.')

        if news_items_list:
            logging.info("✅ News analysis complete.")
            logging.info("--- Overall News Summary ---")
            logging.info(overall_summary)
            logging.info("--------------------------")
            logging.info("--- Individual News Analysis Details ---")
            for i, item in enumerate(news_items_list):
                logging.info(f"News Item {i + 1}:")
                logging.info(f"  Sentiment: {item.get('sentiment', 'N/A')}")
                logging.info(f"  Importance: {item.get('importance', 'N/A'):.2f}")
                logging.info(f"  Explanation: {item.get('explanation', 'No explanation provided.')}")
            logging.info("--------------------------------------")
        else:
            logging.warning(
                "News analysis returned no results or encountered an error. Prediction will not be adjusted by news.")

        news_analysis = news_items_list
    else:
        logging.warning("Could not find any news articles. Prediction will not be adjusted by news.")
        news_analysis = []

    # --- 5. Adjust Prediction with News Sentiment ---
    y_future_adjusted = adjust_prediction_with_news(y_future_raw, news_analysis)
    logging.info("✅ Adjusted prediction based on news sentiment.")

    # --- 6. Plot Results ---
    plt.figure(figsize=(12, 7))
    plt.plot(x_hist, y_hist, 'o-', label="Historical (from Image)", color='royalblue')

    # Connect historical to prediction
    last_hist_x, last_hist_y = x_hist[-1], y_hist[-1]
    first_future_x, first_future_y = x_future[0], y_future_adjusted[0]

    plt.plot([last_hist_x, first_future_x], [last_hist_y, first_future_y], '--', color='gray')
    plt.plot(x_future, y_future_adjusted, 'x--', label="Prediction (Adjusted by News)", color='darkorange',
             markersize=8)

    plt.title(f"{ticker} Stock Price Prediction", fontsize=16)
    plt.xlabel("Time Steps (Relative)", fontsize=12)
    plt.ylabel("Simulated Price ($)", fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the plot
    output_filename = f"{ticker}_prediction_plot.png"
    plt.savefig(output_filename)
    logging.info(f"✅ Plot saved to {output_filename}")

    plt.show()


if __name__ == "__main__":
    # User input interface
    image_path_input = input(
        "Enter the full path to the stock chart image (e.g., /Users/youruser/Desktop/sample_chart.jpg): ")
    ticker_input = input("Enter the stock ticker symbol (e.g., IXIC, AAPL, TSLA): ")

    run_analysis(image_path=image_path_input, ticker=ticker_input)
