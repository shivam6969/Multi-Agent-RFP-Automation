from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import requests
import pdfplumber
import json
import os
import re

# -------------------------------
# KEYWORDS
# -------------------------------
ELECTRICAL_KEYWORDS = {
    "cables": [
        "cable", "control cable", "power cable", "lt cable", "ht cable",
        "xlpe cable", "armoured cable", "instrumentation cable", "flexible cable",
        "eht cable", "underground cable", "ug cable", "aerial bunched cable",
        "abc cable", "fire survival cable", "optical fibre cable", "ofc"
    ],
    "wires": [
        "wire", "house wire", "conductor", "electrical wire", "wiring",
        "copper wire", "aluminum wire", "flexible wire"
    ],
    "transformers": [
        "transformer", "distribution transformer", "power transformer",
        "dtr", "ict", "auto transformer", "dry type transformer",
        "isolation transformer", "rectifier transformer"
    ],
    "fans": [
        "fan", "ceiling fan", "exhaust fan", "industrial fan", "blower",
        "ventilation fan", "axial fan", "centrifugal fan"
    ],
    "led_lights": [
        "led light", "led fitting", "led bulb", "led tube", "led panel",
        "led street light", "led flood light", "led luminaire"
    ],
    "switches": [
        "switch", "electrical switch", "modular switch", "switch socket",
        "circuit breaker", "mcb", "mccb", "vcb", "acb", "isolator",
        "disconnect switch", "load break switch"
    ]
}

# Flatten all keywords
ALL_KEYWORDS = [kw for kws in ELECTRICAL_KEYWORDS.values() for kw in kws]

# -------------------------------
# EXTRACTION FUNCTIONS
# -------------------------------
def extract_pdf_text(pdf_url, filename):
    """Extract text from PDF URL"""
    try:
        # Download PDF
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        # Save temporarily
        temp_path = os.path.join("extracted_data", f"temp_{filename}")
        os.makedirs("extracted_data", exist_ok=True)

        with open(temp_path, 'wb') as f:
            f.write(response.content)

        # Extract text
        text = ""
        with pdfplumber.open(temp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        # Clean up temp file
        os.remove(temp_path)

        return text.strip()

    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None

def analyze_rfp_text(text):
    """Analyze RFP text and extract key information"""
    if not text:
        return {
            'full_text': '',
            'start_time': [],
            'end_time': [],
            'location': [],
            'contacts': [],
            'products': [],
            'quantity': [],
            'financial_info': [],
            'bid_details': [],
            'bid_end_date': [],
            'bid_opening_date': [],
            'organisation_name': [],
            'evaluation_schedules': []
        }

    def _find_matches(patterns, text, group=1):
        matches = []
        for pattern in patterns:
            for match in re.findall(pattern, text, re.IGNORECASE):
                if isinstance(match, tuple):
                    value = match[group - 1] if len(match) >= group else match[0]
                else:
                    value = match
                value = str(value).strip()
                if value:
                    matches.append(value)
        return list(dict.fromkeys(matches))

    def _extract_products_and_quantities(text):
        products = []
        quantities = []
        lines = [line.strip() for line in text.split('
') if line.strip()]

        for i, line in enumerate(lines):
            qty_match = re.search(r'(?:qty|quantity|units?|nos?|pieces?)[:\s]*([0-9,]+)', line, re.IGNORECASE)
            if qty_match:
                quantity = qty_match.group(1).replace(',', '').strip()
                product_text = line
                if i > 0:
                    product_text = f"{lines[i-1]} | {line}"
                products.append(product_text)
                quantities.append(quantity)

        for line in lines:
            if any(keyword in line.lower() for keyword in ALL_KEYWORDS):
                if len(line) > 12 and line not in products:
                    products.append(line)

        return list(dict.fromkeys(products)), list(dict.fromkeys(quantities))

    analysis = {
        'full_text': text,
        'start_time': [],
        'end_time': [],
        'location': [],
        'contacts': [],
        'products': [],
        'quantity': [],
        'financial_info': [],
        'bid_details': [],
        'bid_end_date': [],
        'bid_opening_date': [],
        'organisation_name': [],
        'evaluation_schedules': []
    }

    analysis['start_time'] = _find_matches([
        r'start[\s]+time[:\s]*([^\n\r]+)',
        r'opening[\s]+time[:\s]*([^\n\r]+)',
        r'bid[\s]+opening[\s]+time[:\s]*([^\n\r]+)',
        r'from[:\s]*([0-9]{1,2}:[0-9]{2}\s*(?:am|pm|AM|PM)?)',
        r'starts?[:\s]+([0-9]{1,2}:[0-9]{2}\s*(?:am|pm|AM|PM)?)'
    ], text)

    analysis['end_time'] = _find_matches([
        r'end[\s]+time[:\s]*([^\n\r]+)',
        r'closing[\s]+time[:\s]*([^\n\r]+)',
        r'bid[\s]+closing[\s]+time[:\s]*([^\n\r]+)',
        r'till[\s]+([0-9]{1,2}:[0-9]{2}\s*(?:am|pm|AM|PM)?)',
        r'to[:\s]*([0-9]{1,2}:[0-9]{2}\s*(?:am|pm|AM|PM)?)'
    ], text)

    analysis['location'] = _find_matches([
        r'location[:\s]*([^\n\r]+)',
        r'place[:\s]*([^\n\r]+)',
        r'venue[:\s]*([^\n\r]+)',
        r'address[:\s]*([^\n\r]+)',
        r'office[\s]+at[:\s]*([^\n\r]+)',
        r'located[\s]+at[:\s]*([^\n\r]+)'
    ], text)

    analysis['contacts'] = _find_matches([
        r'(?:contact(?: person| details)?|contact name|contact us|procurement officer|issuer)[:\s]*([^\n\r]+)',
        r'email[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        r'phone[:\s]*([\d+\-()\\s]{7,})',
        r'mobile[:\s]*([\d+\-()\\s]{7,})',
        r'tel[:\s]*([\d+\-()\\s]{7,})'
    ], text)

    analysis['products'], analysis['quantity'] = _extract_products_and_quantities(text)

    analysis['financial_info'] = _find_matches([
        r'estimated cost[:\s]*([^\n\r]+)',
        r'budget[:\s]*([^\n\r]+)',
        r'value[:\s]*([^\n\r]+)',
        r'amount[:\s]*([^\n\r]+)',
        r'price[:\s]*([^\n\r]+)',
        r'cost[:\s]*([^\n\r]+)',
        r'rs\.?\s*[\d,]+(?:\s*(?:lakh|crore|thousand|million)?)',
        r'inr\s*[\d,]+(?:\s*(?:lakh|crore|thousand|million)?)',
        r'₹\s*[\d,]+(?:\s*(?:lakh|crore|thousand|million)?)'
    ], text)

    analysis['bid_details'] = _find_matches([
        r'\b(?:tender number|bid number|tender no|bid no|tender id|rfp number)[:\s]*([^\n\r]+)',
        r'procurement type[:\s]*([^\n\r]+)',
        r'procurement mode[:\s]*([^\n\r]+)'
    ], text)

    for match in re.findall(r'\bGEM/\d{4}/[A-Z]+/\d{4,}\b', text, re.IGNORECASE):
        if match not in analysis['bid_details']:
            analysis['bid_details'].append(match)

    analysis['bid_end_date'] = _find_matches([
        r'bid end date[:\s]*([^\n\r]+)',
        r'bid closing date[:\s]*([^\n\r]+)',
        r'tender closing date[:\s]*([^\n\r]+)',
        r'last date of submission[:\s]*([^\n\r]+)',
        r'deadline[:\s]*([^\n\r]+)',
        r'due date[:\s]*([^\n\r]+)'
    ], text)

    analysis['bid_opening_date'] = _find_matches([
        r'bid opening date[:\s]*([^\n\r]+)',
        r'tender opening date[:\s]*([^\n\r]+)',
        r'date of opening[:\s]*([^\n\r]+)',
        r'opening date[:\s]*([^\n\r]+)'
    ], text)

    analysis['organisation_name'] = _find_matches([
        r'organisation[:\s]*([^\n\r]+)',
        r'organization[:\s]*([^\n\r]+)',
        r'issued by[:\s]*([^\n\r]+)',
        r'published by[:\s]*([^\n\r]+)',
        r'department[:\s]*([^\n\r]+)',
        r'ministry[:\s]*([^\n\r]+)'
    ], text)

    schedule_section_match = re.search(
        r'evaluation schedules?|schedule.*items?|items?.*schedule.*quantity|item.*category.*quantity',
        text, re.IGNORECASE
    )
    if schedule_section_match:
        schedule_start = schedule_section_match.start()
        schedule_text = text[schedule_start:schedule_start + 2000]
        schedule_lines = [line.strip() for line in schedule_text.split('\n') if line.strip()]
        for line in schedule_lines:
            if re.search(r'schedule\s+\d+|item\s+\d+', line, re.IGNORECASE):
                qty_match = re.search(r'([0-9,]+)(?:\s*$|\s*[^\w])', line)
                if qty_match:
                    item_desc = re.sub(r'schedule\s+\d+|item\s+\d+|[0-9,]+\s*$', '', line, flags=re.IGNORECASE).strip()
                    if item_desc and len(item_desc) > 5:
                        analysis['evaluation_schedules'].append({
                            'item': item_desc,
                            'quantity': qty_match.group(1).replace(',', '').strip()
                        })

    return analysis
def save_analysis(analysis, filename, keyword):
    """Save the analysis to a JSON file"""
    os.makedirs("extracted_data", exist_ok=True)

    analysis_data = {
        'filename': filename,
        'keyword': keyword,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'analysis': analysis
    }

    safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
    analysis_file = os.path.join("extracted_data", f"{safe_filename.replace('.pdf', '')}_analysis.json")

    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Analysis saved: {analysis_file}")

def format_rfp_catalog_entry(bid_title, keyword, pdf_url, filename, analysis):
    """Convert extracted analysis to catalog entry format"""
    from datetime import datetime
    
    # Generate unique ID from bid title
    rfp_id = f"rfp_{re.sub(r'[^a-z0-9]', '', bid_title.lower())[:12]}"
    
    return {
        'id': rfp_id,
        'keyword': keyword,
        'bid_title': bid_title,
        'pdf_url': pdf_url,
        'filename': filename,
        'status': 'pending',
        'timestamp': datetime.now().isoformat(),
        'analysis': {
            'start_time': analysis.get('start_time', []),
            'end_time': analysis.get('end_time', []),
            'location': analysis.get('location', []),
            'contacts': analysis.get('contacts', []),
            'products': analysis.get('products', []),
            'quantity': analysis.get('quantity', []),
            'financial_info': analysis.get('financial_info', []),
            'bid_details': analysis.get('bid_details', [])
        }
    }

def save_rfp_catalog(all_rfps):
    """Save all extracted RFPs to catalog JSON file"""
    os.makedirs("extracted_data", exist_ok=True)
    catalog_file = os.path.join("extracted_data", "rfps_catalog.json")
    
    with open(catalog_file, 'w', encoding='utf-8') as f:
        json.dump(all_rfps, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Catalog saved: {catalog_file} ({len(all_rfps)} RFPs)")

# -------------------------------
# CHROME SETUP (no download)
# -------------------------------
options = Options()
options.add_argument("--headless")  # Run in background
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(options=options)
driver.get("https://bidplus.gem.gov.in/all-bids")

wait = WebDriverWait(driver, 20)

# -------------------------------
# MAIN LOOP
# -------------------------------
all_extracted_data = []
all_rfp_catalog = []

for keyword in ALL_KEYWORDS[:20]:  # Test with first 20 keywords
    print(f"\n🔍 Searching for: {keyword}")

    try:
        # Find search box
        search_box = wait.until(
            EC.visibility_of_element_located((By.XPATH, "//input[@placeholder='Enter Keyword']"))
        )

        # Clear previous search
        search_box.clear()
        search_box.click()
        search_box.send_keys(keyword)
        search_box.send_keys(Keys.RETURN)

        print("✅ Search triggered")

        # Wait for results
        wait.until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(text(),'GEM/')]"))
        )

        time.sleep(3)

        # Get bids
        bids = driver.find_elements(By.XPATH, "//a[contains(text(),'GEM/')]")

        print(f"📄 Found {len(bids)} bids")

        # Process first 2 bids for each keyword (to avoid too much processing)
        for i in range(min(2, len(bids))):
            try:
                bid_link = bids[i]
                bid_text = bid_link.text
                print(f"🔗 Processing: {bid_text}")

                # Get the PDF URL (assuming it's the href attribute)
                pdf_url = bid_link.get_attribute('href')

                if pdf_url and 'pdf' in pdf_url.lower():
                    print(f"📥 Extracting from: {pdf_url}")

                    # Extract text from PDF
                    filename = f"{bid_text.replace('/', '_')}.pdf"
                    extracted_text = extract_pdf_text(pdf_url, filename)

                    if extracted_text:
                        # Analyze the text
                        analysis = analyze_rfp_text(extracted_text)

                        # Save analysis
                        save_analysis(analysis, filename, keyword)

                        # Add to collected data
                        all_extracted_data.append({
                            'keyword': keyword,
                            'bid_title': bid_text,
                            'pdf_url': pdf_url,
                            'analysis': analysis
                        })

                        # Add to catalog with proper format
                        catalog_entry = format_rfp_catalog_entry(bid_text, keyword, pdf_url, filename, analysis)
                        all_rfp_catalog.append(catalog_entry)

                        print("✅ Information extracted and saved")
                    else:
                        print("❌ Failed to extract text from PDF")
                else:
                    print("❌ No PDF URL found")

            except Exception as e:
                print(f"❌ Error processing bid: {e}")

    except Exception as e:
        print(f"❌ Search failed for {keyword}: {e}")

    # Small delay before next keyword
    time.sleep(2)

# -------------------------------
# SAVE SUMMARY & CATALOG
# -------------------------------
summary_file = os.path.join("extracted_data", "extraction_summary.json")
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_extractions': len(all_extracted_data),
        'keywords_processed': len(ALL_KEYWORDS[:20]),
        'data': all_extracted_data
    }, f, indent=2, ensure_ascii=False)

# Save catalog
save_rfp_catalog(all_rfp_catalog)

print(f"\n✅ Extraction complete! Processed {len(all_rfp_catalog)} RFPs")
print(f"📊 Summary saved to: {summary_file}")
print(f"📋 Catalog saved to: extracted_data/rfps_catalog.json")
print("🌐 Start the web interface with: python rfp_agent.py")

# -------------------------------
# DONE
# -------------------------------
driver.quit()