🩺 AI Medical Document Processor Pipeline
Project Submission: Backend Interview Assignment
🎯 Executive Summary
This project delivers a robust, multi-step pipeline designed to accurately extract and classify financial data (Total, Paid, Due, Discount) from various medical documents (images via OCR and raw text).
Built on FastAPI, the architecture prioritizes reliability, auditability, and speed. The system processes raw input through normalization and contextual classification, delivering high-confidence, structured JSON suitable for immediate integration or financial analysis. The associated responsive front-end demonstrates the entire pipeline flow visually.
🛠️ 1. Project Architecture and Setup
1.1 Architecture Overview
The system implements a classic 4-Stage ETL (Extract, Transform, Load) Pipeline pattern, ensuring strong data integrity from input to final output.
Stage
Process
Core Technology
Output
1. Extraction
Reads image/text input and extracts raw numeric tokens using EasyOCR.
FastAPI, EasyOCR
Raw Tokens, Currency Hint
2. Normalization
Cleans tokens, removes noise, and resolves numeric formatting/OCR errors to produce accurate float/integer values.
Python Regex/Core Logic
Clean Numeric Amounts
3. Classification
Uses contextual pattern matching around amounts to assign financial labels (paid, due, total_bill, discount).
Python Regex, Contextual Analysis
Classified Amounts
4. Final Output (Load)
Structures all data, calculates derived fields (e.g., calculated Due amount), and includes source provenance.
FastAPI Pydantic Models
Final Structured JSON

1.2 Setup Instructions
To replicate and run the application locally, you must clone the repository and install the dependencies.
Clone the Repository
git clone [https://github.com/Abhisheksen8269/OCR-Pipeline-Demo.git](https://github.com/Abhisheksen8269/OCR-Pipeline-Demo.git)
cd OCR-Pipeline-Demo


Create and Activate Virtual Environment
python3 -m venv final_env
source final_env/bin/activate


Install Dependencies
pip install -r requirements.txt


Run the Server
python main.py

The application is accessible locally at http://0.0.0.0:8000.
<img width="1909" height="1016" alt="Screenshot from 2025-09-28 08-27-30" src="https://github.com/user-attachments/assets/32e157d9-9334-4253-94d3-689f18e801a3" />



💻 2. API Usage Examples (cURL)
These cURL commands demonstrate how to interact with the primary endpoint (/extract-amounts) to test both text input and OCR functionality.
(Replace <NGROK_URL> with your live public link, e.g., https://bereft-kizzie-unawardable.ngrok-free.dev)
Test 1: Full Pipeline with Text Input
(Tests Extraction, Normalization, Classification)
curl -X POST "<NGROK_URL>/extract-amounts" \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "text=Total: 1500 Paid: 1000 Due: 500 Discount: 10%"


Test 2: Full Pipeline with Image (OCR) Input
(Tests Image Handling, EasyOCR Integration, and the full pipeline)
# This assumes you have the test image ('img7.png') in your current directory.
curl -X POST "<NGROK_URL>/extract-amounts" \
-H "Accept: application/json" \
-F "file=@img7.png;type=image/png"


Test 3: Guardrail Check (Error Handling)
(Tests validation and graceful failure on missing input)
curl -X POST "<NGROK_URL>/extract-amounts" \
-H "Content-Type: application/x-www-form-urlencoded" \
-d ""


💾 3. Submission Confirmation
Working Backend Demo: Deployed live at: <NGROK_URL> (The link you used: https://bereft-kizzie-unawardable.ngrok-free.dev)
Source Code: Provided via this GitHub Repository.
Screen Recording: Submitted video confirms all endpoints and the responsive UI are working correctly.

