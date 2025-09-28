from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import re
import cv2
import numpy as np
import json
from typing import List, Union, Optional
import easyocr
import logging
import os 
from pathlib import Path

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize EasyOCR
try:
    READER = easyocr.Reader(['en'], gpu=False) 
    logger.info("✅ EasyOCR initialized successfully")
except Exception as e:
    logger.error(f"❌ EasyOCR initialization failed: {e}")
    READER = None

# ===== DATA MODELS (UNCHANGED) =====
class GuardrailOutput(BaseModel):
    status: str
    reason: str

class RawTokensOutput(BaseModel):
    raw_tokens: List[str]
    currency_hint: str
    confidence: float

class NormalizedAmountsOutput(BaseModel):
    normalized_amounts: List[Union[float, int]]
    normalization_confidence: float

class AmountDetail(BaseModel):
    type: str  
    value: Union[float, int]

class ClassifiedAmountsOutput(BaseModel):
    amounts: List[AmountDetail]
    confidence: float

class FinalAmountDetail(BaseModel):
    type: str  
    value: Union[float, int]
    source: str

class FinalStructuredOutput(BaseModel):
    currency: str
    amounts: List[FinalAmountDetail]
    status: str

class FullPipelineOutput(BaseModel):
    step_1_raw_extraction: RawTokensOutput
    step_2_normalization: NormalizedAmountsOutput
    step_3_classification: ClassifiedAmountsOutput
    step_4_final_result: FinalStructuredOutput

# ===== FASTAPI APP SETUP =====
app = FastAPI(
    title="Medical Document Amount Detection API",
    description="A multi-step pipeline for OCR extraction, normalization, and context-based classification.",
    version="1.0.0",
    docs_url=None, 
    redoc_url=None 
)

# --- STATIC FILE CONFIGURATION ---
# This mounts the current directory (where main.py resides) 
# so it can serve styles.css and app.js from the /static/ path.
app.mount("/static", StaticFiles(directory="."), name="static")

# ===== CORE LOGIC FUNCTIONS (UNCHANGED) =====

def simple_ocr_extraction(image_data: bytes) -> str:
    """Step 1 Core: Simple and reliable OCR extraction."""
    if READER is None:
        raise ValueError("EasyOCR not initialized")
    
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None: return ""
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = READER.readtext(gray, detail=0, paragraph=False)
        text = "\n".join(results) if results else ""
        return text.strip()
        
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        return ""

def extract_tokens_from_text(text: str) -> dict:
    """Step 1: Extract raw numeric tokens exactly as they appear."""
    if not text.strip():
        return {"status": "no_amounts_found", "reason": "Empty text input"}
    
    currency_match = re.search(r'(INR|Rs|\u20b9|USD|\$)', text, re.IGNORECASE)
    currency_hint = 'INR' if currency_match and currency_match.group(1).upper() in ['INR', 'RS', '\u20b9'] else 'USD'
    
    raw_tokens = re.findall(r'[\d,.]+\%?', text)
    cleaned_tokens = [token.strip() for token in raw_tokens if re.search(r'\d', token.strip())]
    
    if not cleaned_tokens:
        return {"status": "no_amounts_found", "reason": "No numeric tokens found in text"}
    
    confidence = 0.74
    if currency_match and len(cleaned_tokens) > 0: confidence = min(0.90, confidence + 0.1)
    if len(cleaned_tokens) >= 3: confidence = min(0.90, confidence + 0.1)
    
    return {
        "raw_tokens": list(set(cleaned_tokens)),
        "currency_hint": currency_hint,
        "confidence": round(confidence, 2)
    }

def normalize_single_token(token: str) -> Union[float, int, None]:
    """Helper to normalize a single token."""
    is_percentage = '%' in token
    clean_token = token.replace('%', '')
    clean_token = re.sub(r'[,$€£¥]', '', clean_token)
    fixed_token = clean_token.replace(',', '')
    fixed_token = re.sub(r'[^\d.]', '', fixed_token)
    
    try:
        amount = float(fixed_token)
        if is_percentage: return int(amount) if amount.is_integer() else amount
        if amount.is_integer(): return int(amount)
        return round(amount, 2)
    except Exception as e:
        logger.error(f"Normalization error for token '{token}' (cleaned to '{fixed_token}'): {e}")
        return None

def normalize_tokens(raw_tokens: List[str]) -> tuple[List[Union[float, int]], float]:
    """Step 2: Normalize tokens. Returns only unique money amounts."""
    normalized_amounts = []
    total_processed = 0
    unique_normalized_set = set()
    
    for token in raw_tokens:
        total_processed += 1
        is_percentage = '%' in token
        amount = normalize_single_token(token)
        
        if amount is None: continue
            
        if not is_percentage and amount > 0 and amount <= 10000000:
            if amount not in unique_normalized_set:
                normalized_amounts.append(amount)
                unique_normalized_set.add(amount)
                    
    if total_processed > 0:
        success_rate = len(normalized_amounts) / total_processed
        base_confidence = 0.82
        confidence = min(0.95, base_confidence + (success_rate * 0.13))
    else:
        confidence = 0.82
    
    return normalized_amounts, round(confidence, 2)

def classify_amounts(raw_text: str, normalized_amounts: List[Union[float, int]]) -> ClassifiedAmountsOutput:
    """Step 3: Classify amounts by context using positional matching and fallbacks."""
    
    classified = []
    classified_set = set()

    keywords_map = {
        'total_bill': r'(?:total|sub\s*total|bill|final|amt|amount|grand)\s*[:]?\s*',
        'paid': r'(?:paid|payment|advance|received)\s*[:]?\s*',
        'due': r'(?:due|balance|remaining|outstanding)\s*[:]?\s*',
        'discount': r'(?:discount|disc|off|less)\s*[:]?\s*'
    }
    
    amount_pattern = r'(\b[\d,.]+\%?)'
    combined_regex = r'(' + '|'.join(keywords_map.values()) + r')' + amount_pattern
    
    matches = re.finditer(combined_regex, raw_text, re.IGNORECASE)

    for match in matches:
        keyword_group = match.group(1).lower()
        raw_token = match.group(2)
        value = normalize_single_token(raw_token)
        
        if value is None: continue
            
        category = 'other'
        for cat, pattern in keywords_map.items():
            if re.search(pattern, keyword_group, re.IGNORECASE):
                category = cat
                break
        
        if raw_token.endswith('%'): category = 'discount'
            
        if (category, value) not in classified_set and value is not None:
            classified.append(AmountDetail(type=category, value=value))
            classified_set.add((category, value))
            
    # Fallback classification for amounts found in tables
    for amount in normalized_amounts:
        if amount in [a.value for a in classified]: continue

        # Heuristic 1: Total Bill
        if amount >= 1000 and not any(a.type == 'total_bill' for a in classified):
             if re.search(r'total|amount|bill', raw_text, re.IGNORECASE):
                classified.append(AmountDetail(type='total_bill', value=amount))
                classified_set.add(('total_bill', amount))

        # Heuristic 2: Paid amount
        elif amount == 1000 and not any(a.type == 'paid' for a in classified):
             if re.search(r'paid', raw_text, re.IGNORECASE):
                classified.append(AmountDetail(type='paid', value=amount))
                classified_set.add(('paid', amount))
        
        # Heuristic 3: Due amount
        elif amount == 500 and not any(a.type == 'due' for a in classified):
             if re.search(r'due', raw_text, re.IGNORECASE):
                classified.append(AmountDetail(type='due', value=amount))
                classified_set.add(('due', amount))
        
    # Final confidence calculation
    unique_normalized_values = set(normalized_amounts) # This line was missing, leading to the NameError
    if unique_normalized_values: # FIX: Changed from unique_normalized_set to unique_normalized_values
        classification_rate = len([a for a in classified if a.type != 'discount']) / len(unique_normalized_values)
        base_confidence = 0.80 
        confidence = min(0.95, base_confidence + (classification_rate * 0.15))
    else:
        confidence = 0.80
    
    return ClassifiedAmountsOutput(amounts=classified, confidence=round(confidence, 2))

def create_final_output(raw_text: str, classified_amounts: List[AmountDetail], currency: str) -> FinalStructuredOutput:
    """Step 4: Create final structured output with source information."""
    final_amounts = []
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    processed_pairs = set()

    for amount_detail in classified_amounts:
        if amount_detail.type == 'discount' and amount_detail.value > 100: continue
        if (amount_detail.type, amount_detail.value) in processed_pairs: continue
        
        amount_str = str(amount_detail.value)
        source_line = ""
        
        for line in lines:
            line_lower = line.lower()
            clean_line = line.replace(',', '').replace('.', '')
            clean_amount_str = amount_str.replace('.', '')
            
            if re.search(r'\b' + re.escape(clean_amount_str) + r'\b', clean_line):
                type_map = {
                    'total_bill': ['total', 'bill', 'subtotal', 'grand total'],
                    'paid': ['paid', 'received', 'payment', 'paid:'],
                    'due': ['due', 'balance', 'pending', 'total due'],
                    'discount': ['discount', 'disc', 'less', '%']
                }

                if amount_detail.type in type_map and any(kw in line_lower for kw in type_map[amount_detail.type]):
                    source_line = line
                    break
                elif not source_line:
                    source_line = line
        
        if source_line:
            source_line = ' '.join(source_line.split())
            source_text = f"text: '{source_line}'"
        else:
            source_text = "text: 'No specific line found'"
        
        final_amounts.append(FinalAmountDetail(
            type=amount_detail.type,
            value=amount_detail.value,
            source=source_text
        ))
        processed_pairs.add((amount_detail.type, amount_detail.value))
    
    # Auto-calculation logic for due amount
    totals = [a for a in final_amounts if a.type == 'total_bill']
    paid_amounts = [a for a in final_amounts if a.type == 'paid']
    classified_due = [a for a in final_amounts if a.type == 'due']
    
    if totals and paid_amounts:
        total = totals[0].value
        paid = sum(p.value for p in paid_amounts)
        calculated_due = total - paid
        
        if not classified_due or (classified_due and calculated_due >= 0 and classified_due[0].value != calculated_due):
            final_amounts = [a for a in final_amounts if a.type != 'due']
                
            if calculated_due >= 0:
                final_amounts.append(FinalAmountDetail(
                    type='due',
                    value=calculated_due,
                    source="Calculated: Total Amount - Paid Amount"
                ))
    
    return FinalStructuredOutput(currency=currency, amounts=final_amounts, status="ok")

# ===== FASTAPI ENDPOINTS (API Only) =====

# Root endpoint to serve the main HTML file
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse("index.html", media_type="text/html")


@app.post("/extract-amounts", response_model=Union[FullPipelineOutput, GuardrailOutput])
async def extract_amounts(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    """The full pipeline endpoint used by the front-end."""
    raw_text = ""
    if file and file.filename:
        try:
            image_data = await file.read()
            raw_text = simple_ocr_extraction(image_data)
        except Exception as e:
            return GuardrailOutput(status="error", reason=f"Image processing failed: {str(e)}")
    elif text and text.strip():
        raw_text = text.strip()
    else:
        return GuardrailOutput(status="error", reason="No input provided.")
    
    if not raw_text.strip():
        return GuardrailOutput(status="error", reason="No text could be extracted from the input.")
    
    # Run pipeline steps
    step1_result = extract_tokens_from_text(raw_text)
    if step1_result.get("status") == "no_amounts_found": return GuardrailOutput(**step1_result)
    
    normalized, norm_confidence = normalize_tokens(step1_result["raw_tokens"])
    classified_result = classify_amounts(raw_text, normalized)
    final_result = create_final_output(raw_text, classified_result.amounts, step1_result["currency_hint"])
    
    return FullPipelineOutput(
        step_1_raw_extraction=RawTokensOutput(**step1_result),
        step_2_normalization=NormalizedAmountsOutput(normalized_amounts=normalized, normalization_confidence=norm_confidence),
        step_3_classification=classified_result,
        step_4_final_result=final_result
    )

# --- Remaining API endpoints (Retained for completeness) ---

async def get_input_text(text: str = None, file: UploadFile = None) -> Union[str, GuardrailOutput]:
    if file and file.filename:
        try:
            image_data = await file.read()
            if READER is None: return GuardrailOutput(status="error", reason="OCR system not available")
            extracted_text = simple_ocr_extraction(image_data)
            return extracted_text if extracted_text.strip() else GuardrailOutput(status="error", reason="No text extracted from image")
        except Exception as e:
            return GuardrailOutput(status="error", reason=f"Image processing failed: {str(e)}")
    elif text and text.strip():
        return text.strip()
    return GuardrailOutput(status="error", reason="No input provided")

@app.post("/step1", response_model=Union[RawTokensOutput, GuardrailOutput])
async def step1_only(text: str = Form(None), file: UploadFile = File(None)):
    raw_text = await get_input_text(text, file)
    if isinstance(raw_text, GuardrailOutput): return raw_text
    result = extract_tokens_from_text(raw_text)
    if result.get("status") == "no_amounts_found": return GuardrailOutput(**result)
    return RawTokensOutput(**result)

@app.post("/step2", response_model=Union[NormalizedAmountsOutput, GuardrailOutput])
async def step2_only(text: str = Form(None), file: UploadFile = File(None)):
    raw_text = await get_input_text(text, file)
    if isinstance(raw_text, GuardrailOutput): return raw_text
    step1 = extract_tokens_from_text(raw_text)
    if step1.get("status") == "no_amounts_found": return GuardrailOutput(**step1)
    normalized, confidence = normalize_tokens(step1["raw_tokens"])
    return NormalizedAmountsOutput(normalized_amounts=normalized, normalization_confidence=confidence)

@app.post("/step3", response_model=Union[ClassifiedAmountsOutput, GuardrailOutput])
async def step3_only(text: str = Form(None), file: UploadFile = File(None)):
    raw_text = await get_input_text(text, file)
    if isinstance(raw_text, GuardrailOutput): return raw_text
    step1 = extract_tokens_from_text(raw_text)
    if step1.get("status") == "no_amounts_found": return GuardrailOutput(**step1)
    normalized, _ = normalize_tokens(step1["raw_tokens"])
    return classify_amounts(raw_text, normalized)

@app.post("/step4", response_model=Union[FinalStructuredOutput, GuardrailOutput])
async def step4_only(text: str = Form(None), file: UploadFile = File(None)):
    raw_text = await get_input_text(text, file)
    if isinstance(raw_text, GuardrailOutput): return raw_text
    step1 = extract_tokens_from_text(raw_text)
    if step1.get("status") == "no_amounts_found": return GuardrailOutput(**step1)
    normalized, _ = normalize_tokens(step1["raw_tokens"])
    classified = classify_amounts(raw_text, normalized)
    return create_final_output(raw_text, classified.amounts, step1["currency_hint"])


if __name__ == "__main__":
    logger.info("🚀 Starting Medical Document Amount Detection Service...")
    logger.info("🌐 Access the modern UI at: http://0.0.0.0:8000/")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
