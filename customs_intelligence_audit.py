# ============================================
# 1. SETUP & INSTALLATION
# ============================================
# pip install -q -U google-genai ultralytics
import os
from google import genai
from google.genai import types
from ultralytics import YOLO

# ============================================
# 2. CONFIGURATION (SAFE)
# ============================================
# IMPORTANT:
# Set your API key as an environment variable:
# export GOOGLE_API_KEY="your_key_here"
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("❌ ERROR: Missing API key. Set GOOGLE_API_KEY as an environment variable.")

client = genai.Client(api_key=API_KEY)
STRATEGIC_MODEL = "gemini-3.1-flash-lite-preview"

# ============================================
# 3. DOCUMENT PATHS (UPDATED)
# ============================================
# DOCUMENT TYPES USED IN THIS AUDIT:
# 1. Bill of Lading (BOL) – Shipment and consignee details.
# 2. Commercial Invoice – Declared value, quantity, and item description.
# 3. PAAR (Pre‑Arrival Assessment Report) – 
#    A Nigeria Customs Service risk‑assessment document containing HS codes,
#    valuation, and clearance recommendations.
# 4. Packing List – Breakdown of physical contents and quantities.

doc_list = [
    '/content/Bill_of_Lading.pdf',
    '/content/Commercial_Invoice.pdf',
    '/content/PAAR_Document.pdf',
    '/content/Packing_List.pdf'
]

# X‑RAY IMAGE TO BE DETECTED
cargo_img = '/content/xray_image.jpeg'

# YOLOv8 BEST WEIGHT FILE
weights_path = '/content/best.pt'


# ============================================
# 4. MAIN CUSTOMS INTELLIGENCE AUDIT FUNCTION
# ============================================
def run_customs_intelligence_audit(document_paths, image_path):
    # --- SAFETY CHECK ---
    for p in document_paths + [image_path, weights_path]:
        if not os.path.exists(p):
            print(f"❌ STOP: File not found at {p}. Please upload or fix the path.")
            return

    # --- PHASE 1: MULTI-DOCUMENT EXTRACTION ---
    print("Phase 1: Extraction (Reading BOL, Invoice, PAAR, Packing List)...")
    parts = []
    for path in document_paths:
        with open(path, "rb") as f:
            file_bytes = f.read()

        mime = 'application/pdf' if path.lower().endswith('.pdf') else 'image/jpeg'
        parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime))

    doc_prompt = """
    Analyze these Customs documents. Extract:
    1. Consignee Name.
    2. Declared Goods Description.
    3. All HS Codes (Harmonized System Codes).
    4. Total Quantity/Weight.
    Provide a clear summary of what SHOULD be in the container.
    """

    manifest_result = client.models.generate_content(
        model=STRATEGIC_MODEL,
        contents=parts + [doc_prompt]
    )

    expected_data = manifest_result.text.strip()
    print(f"\n--- EXPECTED DATA (From Documents) ---\n{expected_data}")

    # --- PHASE 2: PHYSICAL VISION (YOLOv8) ---
    print("\nPhase 2: Vision (Scanning Cargo Image)...")
    vision_model = YOLO(weights_path)
    results = vision_model.predict(image_path, conf=0.20)

    found_names = [vision_model.names[int(box.cls)] for box in results[0].boxes]
    detected_items = ", ".join(found_names) if found_names else "nothing detected"
    print(f"Vision Detection Result: {detected_items}")

    # --- PHASE 3: CONFORMITY & RISK AUDIT ---
    print("\nPhase 3: Audit (Cross-Referencing Documents vs. Vision)...")
    audit_prompt = f"""
    You are an expert Nigeria Customs Service Auditor.

    DATA 1 (Documents/HS Codes): {expected_data}
    DATA 2 (YOLO Vision Detections): {detected_items}

    TASK:
    - Compare the items detected by vision with the HS codes in the documents.
    - Identify any mismatch between declared goods and physical cargo.
    - Status: [CONFORMITY or NON-CONFORMITY]
    - Reasoning: Explain any under-valuation, misclassification, or concealment risks.
    - Risk Level: High, Medium, or Low.
    """

    audit_report = client.models.generate_content(
        model=STRATEGIC_MODEL,
        contents=audit_prompt
    )

    print("\n" + "="*50)
    print("FINAL CUSTOMS INTELLIGENCE AUDIT REPORT")
    print("="*50)
    print(audit_report.text)


# ============================================
# 5. EXECUTE
# ============================================
run_customs_intelligence_audit(doc_list, cargo_img)
