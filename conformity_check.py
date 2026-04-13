# 1. SETUP
# pip install -q -U google-genai ultralytics
from google import genai
from ultralytics import YOLO
import os

# 2. CONFIGURATION
# IMPORTANT:
# Store your API key as an environment variable:
# export GOOGLE_API_KEY="your_key_here"
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("Missing API key. Set GOOGLE_API_KEY as an environment variable.")

STRATEGIC_MODEL = "gemini-3.1-flash-lite-preview"

# Define expected manifest item
EXPECTED_ITEMS = "truck tyre"  # Change based on your manifest

client = genai.Client(api_key=API_KEY)

def run_conformity_check(image_path, expected):
    # --- VISION: YOLOv8 ---
    vision_model = YOLO('/content/drive/MyDrive/YOLOv8_Project/my_custom_training/weights/best.pt')
    results = vision_model.predict(image_path, conf=0.25)

    # Extract detections
    names = vision_model.names
    found = [names[int(box.cls)] for box in results[0].boxes]
    summary = ", ".join(found) if found else "nothing"

    print(f"Expected: {expected}")
    print(f"Detected: {summary}")

    # --- REASONING: LLM CONFORMITY TASK ---
    prompt = f"""
    SYSTEM: You are a Quality Control & Customs Auditor.

    TASK: Compare the 'EXPECTED' manifest item with the 'DETECTED' vision results.
    - If the detected items match the expected item, declare 'CONFORMITY'.
    - If they do not match, or if something extra/illegal is found, declare 'NON-CONFORMITY' and explain why.

    DATA:
    - Expected Item: {expected}
    - YOLOv8 Detections: {summary}

    RESPONSE FORMAT:
    1. Status: [CONFORMITY or NON-CONFORMITY]
    2. Reasoning: (Brief explanation)
    """

    try:
        response = client.models.generate_content(
            model=STRATEGIC_MODEL,
            contents=prompt
        )

        print("\n" + "="*45)
        print("CONFORMITY AUDIT REPORT")
        print("="*45)
        print(response.text)

    except Exception as e:
        print(f"API Error: {e}")

# 4. RUN
test_image = '/content/drive/MyDrive/ncs x-ray images/hides and skin.jpeg'
run_conformity_check(test_image, EXPECTED_ITEMS)
