import streamlit as st
import pdfplumber
import requests
import json
import os
import re
import html
import time
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import tempfile

# Page configuration
st.set_page_config(
    page_title="RFP Scraping Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rfp_sessions' not in st.session_state:
    st.session_state.rfp_sessions = []

if 'current_session' not in st.session_state:
    st.session_state.current_session = None

if 'agent_running' not in st.session_state:
    st.session_state.agent_running = False

# Keywords
ELECTRICAL_KEYWORDS = {
    "switches": [
        "switch", "electrical switch", "modular switch", "switch socket",
        "circuit breaker", "mcb", "mccb", "vcb", "acb", "isolator",
        "disconnect switch", "load break switch"
    ],
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
    ]
}
ALL_KEYWORDS = [kw for kws in ELECTRICAL_KEYWORDS.values() for kw in kws]

def sanitize_text(value, length=None):
    text = re.sub(r'<[^>]+>', '', str(value))
    text = re.sub(r'\s+', ' ', text).strip()
    if length and len(text) > length:
        text = text[: length - 3].rstrip() + '...'
    return html.escape(text)


def inject_card_styles():
    st.markdown(
        """
        <style>
        .rfp-card {
            width: 100%;
            min-width: 0;
            max-width: 320px;
            border-radius: 22px;
            background: #ffffff;
            padding: 20px;
            margin-bottom: 14px;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.18);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .rfp-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }
        .rfp-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 700;
            color: #2563eb;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            background: rgba(59, 130, 246, 0.08);
            padding: 8px 12px;
            border-radius: 999px;
        }
        .rfp-date {
            text-align: right;
        }
        .rfp-date-month {
            font-size: 11px;
            color: #64748b;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .rfp-date-day {
            font-size: 26px;
            font-weight: 800;
            color: #0f172a;
            line-height: 1;
        }
        .rfp-title {
            font-size: 15px;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.35;
            margin-bottom: 14px;
            min-height: 56px;
        }
        .rfp-details {
            font-size: 13px;
            color: #475569;
            line-height: 1.65;
            margin-bottom: 18px;
        }
        .rfp-details .rfp-field {
            margin-bottom: 10px;
        }
        .rfp-details strong {
            color: #0f172a;
            font-weight: 700;
        }
        .rfp-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            padding: 12px 14px;
            border-radius: 16px;
            background: rgba(248, 250, 252, 0.95);
            font-size: 12px;
            color: #475569;
        }
        .rfp-footer .status {
            font-weight: 700;
        }
        .rfp-card-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def extract_pdf_text(pdf_url, filename):
    """Extract text from PDF URL"""
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name

        text = ""
        with pdfplumber.open(temp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        os.unlink(temp_path)
        return text.strip()

    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None


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
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    for i, line in enumerate(lines):
        qty_match = re.search(r'\b(?:qty|quantity|units?|nos?|pieces?)[:\s]*([0-9,]+)\b', line, re.IGNORECASE)
        if qty_match:
            quantity = qty_match.group(1).replace(',', '').strip()
            product_text = line
            if i > 0 and len(lines[i - 1]) > 10:
                product_text = f"{lines[i-1]} | {line}"
            product_text = sanitize_text(product_text, 120)
            products.append(product_text)
            quantities.append(quantity)

    for line in lines:
        if any(keyword in line.lower() for keyword in ALL_KEYWORDS):
            candidate = sanitize_text(line, 120)
            if len(candidate) > 12 and candidate not in products:
                products.append(candidate)

    return list(dict.fromkeys(products)), list(dict.fromkeys(quantities))


def load_stock_catalog():
    stock_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stock_company'))
    stock_items = []

    if not os.path.isdir(stock_dir):
        return stock_items

    for filename in os.listdir(stock_dir):
        if not filename.lower().endswith('.json'):
            continue

        filepath = os.path.join(stock_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        for entry in data.get('product_data', []):
            range_name = entry.get('range', '')
            available = entry.get('monthly_avalaible') or entry.get('monthly_available')

            if 'items' in entry:
                for item in entry.get('items', []):
                    stock_items.append({
                        'item_name': item.get('item_name', ''),
                        'range': range_name,
                        'color': item.get('color', '') or '',
                        'sap_code': item.get('sap_code', ''),
                        'available': available
                    })

            if 'colors' in entry:
                for color_entry in entry.get('colors', []):
                    color_name = color_entry.get('color', '')
                    for item in color_entry.get('items', []):
                        stock_items.append({
                            'item_name': item.get('item_name', ''),
                            'range': range_name,
                            'color': color_name,
                            'sap_code': item.get('sap_code', ''),
                            'available': available
                        })

    return stock_items


def _normalize_text(text):
    return re.sub(r'[^a-z0-9]+', ' ', str(text).lower()).strip()


def _parse_quantity(quantity_text):
    if not quantity_text:
        return None
    quantity_text = quantity_text.replace(',', '')
    match = re.search(r'(\d+)', quantity_text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def compare_rfp_with_stock(rfp_analysis, stock_items):
    products = rfp_analysis.get('products', [])
    quantities = rfp_analysis.get('quantity', [])
    parsed_quantities = [_parse_quantity(q) for q in quantities]
    stock_matches = []
    no_match_products = []

    for idx, product in enumerate(products):
        requested_qty = parsed_quantities[idx] if idx < len(parsed_quantities) else (parsed_quantities[0] if parsed_quantities else None)
        product_text = _normalize_text(product)
        if not product_text:
            continue

        candidates = []
        product_tokens = set(product_text.split())

        for item in stock_items:
            item_text = _normalize_text(f"{item.get('item_name', '')} {item.get('range', '')} {item.get('color', '')}")
            sap_code = str(item.get('sap_code', '')).lower()

            if sap_code and sap_code in product_text:
                candidates.append((item, 100))
                continue

            item_tokens = set(item_text.split())
            if not item_tokens:
                continue

            common_tokens = product_tokens & item_tokens
            score = len(common_tokens)
            if score >= 3:
                candidates.append((item, score))
                continue

            if item_text and (item_text in product_text or product_text in item_text):
                candidates.append((item, score + 1))

        if candidates:
            best_item = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
            available = best_item.get('available')
            enough = None
            if requested_qty is not None and isinstance(available, int):
                enough = requested_qty <= available
            stock_matches.append({
                'product': product,
                'requested_qty': requested_qty,
                'matched_item': best_item,
                'available': available,
                'enough': enough
            })
        else:
            no_match_products.append({'product': product, 'requested_qty': requested_qty})

    return {
        'matches': stock_matches,
        'no_matches': no_match_products
    }


STOCK_ITEMS = load_stock_catalog()


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
def run_rfp_scraping_agent(keywords_limit=3, bids_per_keyword=2):
    """Run the automated RFP scraping agent"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=options)
    driver.get("https://bidplus.gem.gov.in/all-bids")

    wait = WebDriverWait(driver, 20)

    session_data = {
        'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'start_time': datetime.now().isoformat(),
        'rfps': [],
        'status': 'running'
    }

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        keywords_to_process = ALL_KEYWORDS[:keywords_limit]

        for i, keyword in enumerate(keywords_to_process):
            status_text.text(f"🔍 Searching for: {keyword} ({i+1}/{len(keywords_to_process)})")

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

                # Wait for results
                wait.until(
                    EC.presence_of_element_located((By.XPATH, "//a[contains(text(),'GEM/')]"))
                )

                time.sleep(2)

                # Get bids
                bids = driver.find_elements(By.XPATH, "//a[contains(text(),'GEM/')]")

                for j in range(min(bids_per_keyword, len(bids))):
                    bid_link = bids[j]
                    bid_text = bid_link.text

                    # Get PDF URL
                    pdf_url = bid_link.get_attribute('href')

                    if pdf_url and 'pdf' in pdf_url.lower():
                        # Extract text
                        filename = f"{bid_text.replace('/', '_')}.pdf"
                        extracted_text = extract_pdf_text(pdf_url, filename)

                        if extracted_text:
                            analysis = analyze_rfp_text(extracted_text)

                            rfp_data = {
                                'id': f"{session_data['session_id']}_{i}_{j}",
                                'keyword': keyword,
                                'bid_title': bid_text,
                                'pdf_url': pdf_url,
                                'filename': filename,
                                'analysis': analysis,
                                'status': 'pending',  # pending, move_forward, hold
                                'timestamp': datetime.now().isoformat()
                            }

                            session_data['rfps'].append(rfp_data)

            except Exception as e:
                st.error(f"Error processing keyword {keyword}: {e}")

            # Update progress
            progress = (i + 1) / len(keywords_to_process)
            progress_bar.progress(progress)

        session_data['status'] = 'completed'
        session_data['end_time'] = datetime.now().isoformat()

    finally:
        driver.quit()

    progress_bar.empty()
    status_text.empty()

    return session_data

def save_session(session_data):
    """Save session data to file"""
    os.makedirs("sessions", exist_ok=True)
    session_file = os.path.join("sessions", f"session_{session_data['session_id']}.json")

    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    return session_file

def load_sessions():
    """Load all saved sessions"""
    sessions = []
    if os.path.exists("sessions"):
        for file in os.listdir("sessions"):
            if file.startswith("session_") and file.endswith(".json"):
                with open(os.path.join("sessions", file), 'r') as f:
                    sessions.append(json.load(f))
    return sorted(sessions, key=lambda x: x['start_time'], reverse=True)

def load_rfp_catalog():
    """Load RFP catalog from extracted_data/rfps_catalog.json"""
    catalog_path = os.path.join("..", "extracted_data", "rfps_catalog.json")
    if os.path.exists(catalog_path):
        with open(catalog_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def initialize_session_from_catalog():
    """Create a session from the RFP catalog if no sessions exist"""
    rfp_catalog = load_rfp_catalog()
    if not rfp_catalog:
        return None
    
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_data = {
        'session_id': session_id,
        'start_time': datetime.now().isoformat(),
        'rfps': rfp_catalog  # Catalog RFPs already have the correct structure
    }
    
    # Save this session
    save_session(session_data)
    return session_data

def display_rfp_card(rfp, session_id, parent=None):
    """Display an RFP as a structured session card"""
    container = parent if parent is not None else st

    location = sanitize_text(rfp['analysis']['location'][0] if rfp['analysis']['location'] else 'N/A', 28)
    contact = sanitize_text(rfp['analysis']['contacts'][0] if rfp['analysis']['contacts'] else 'N/A', 24)
    product = sanitize_text(rfp['analysis']['products'][0] if rfp['analysis']['products'] else 'N/A', 32)
    quantity = sanitize_text(rfp['analysis']['quantity'][0] if rfp['analysis']['quantity'] else 'N/A', 16)
    financial = sanitize_text(rfp['analysis']['financial_info'][0] if rfp['analysis']['financial_info'] else 'N/A', 18)
    bid_title = sanitize_text(rfp['bid_title'], 72)
    bid_end_date = sanitize_text(rfp['analysis'].get('bid_end_date', ['N/A'])[0] if rfp['analysis'].get('bid_end_date', []) else 'N/A', 24)
    bid_opening_date = sanitize_text(rfp['analysis'].get('bid_opening_date', ['N/A'])[0] if rfp['analysis'].get('bid_opening_date', []) else 'N/A', 24)
    organisation_name = sanitize_text(rfp['analysis'].get('organisation_name', ['N/A'])[0] if rfp['analysis'].get('organisation_name', []) else 'N/A', 28)

    try:
        timestamp_dt = datetime.fromisoformat(rfp['timestamp'])
        date_day = timestamp_dt.strftime('%d')
        date_month = timestamp_dt.strftime('%b').upper()
    except Exception:
        date_day = '--'
        date_month = 'N/A'

    status_color = '#F59E0B' if rfp['status'] == 'pending' else '#16A34A' if rfp['status'] == 'move_forward' else '#DC2626'
    status_emoji = '⏳' if rfp['status'] == 'pending' else '✅' if rfp['status'] == 'move_forward' else '⏸️'
    status_label = rfp['status'].replace('_', ' ').title()

    card_html = f"""
    <div class='rfp-card'>
      <div class='rfp-card-header'>
        <div class='rfp-badge'>RFP</div>
        <div class='rfp-date'>
          <div class='rfp-date-month'>{date_month}</div>
          <div class='rfp-date-day'>{date_day}</div>
        </div>
      </div>

      <div class='rfp-title'>{bid_title}</div>

      <div class='rfp-details'>
        <div class='rfp-field'><strong>📍 Place:</strong> {location}</div>
        <div class='rfp-field'><strong>🏢 Organisation:</strong> {organisation_name}</div>
        <div class='rfp-field'><strong>📅 Bid End:</strong> {bid_end_date}</div>
        <div class='rfp-field'><strong>📂 Bid Opening:</strong> {bid_opening_date}</div>
        <div class='rfp-field'><strong>📞 Contact:</strong> {contact}</div>
        <div class='rfp-field'><strong>📦 Product:</strong> {product}</div>
        <div class='rfp-field'><strong>📊 Qty:</strong> {quantity}</div>
        <div class='rfp-field'><strong>💰 Budget:</strong> {financial}</div>
      </div>

      <div class='rfp-footer'>
        <span>Session {session_id[-6:]}</span>
        <span class='status' style='color:{status_color};'>{status_emoji} {status_label}</span>
      </div>
    </div>
    """

    container.markdown(card_html, unsafe_allow_html=True)
    
    # Display evaluation schedules table if available
    schedules = rfp['analysis'].get('evaluation_schedules', [])
    if schedules:
        container.markdown("**📋 Evaluation Schedules:**", unsafe_allow_html=True)
        schedule_data = []
        for sched in schedules:
            schedule_data.append({
                'Item/Category': sched.get('item', 'N/A'),
                'Quantity': sched.get('quantity', 'N/A')
            })
        df = pd.DataFrame(schedule_data)
        container.dataframe(df, use_container_width=True, hide_index=True)

    stock_match = rfp.get('stock_match')
    if stock_match and rfp['status'] == 'move_forward':
        container.markdown("**✅ Stock Comparison Result:**")
        if stock_match.get('matches'):
            for match in stock_match['matches']:
                item = match['matched_item']
                enough = match['enough']
                status_text = 'Enough available' if enough else 'Not enough available' if enough is False else 'Availability unknown'
                product_display = sanitize_text(match['product'], 80)
                item_name = sanitize_text(item.get('item_name', 'N/A'), 40)
                item_range = sanitize_text(item.get('range', 'N/A'), 40)
                container.write(f"• {product_display} → {item_name} ({item_range}) — requested {match.get('requested_qty', 'N/A')}, available {match.get('available', 'N/A')} — {status_text}")
        if stock_match.get('no_matches'):
            no_match_items = [sanitize_text(p['product'], 80) for p in stock_match['no_matches']]
            container.write("• No stock catalog match found for: " + ", ".join(no_match_items))

    if rfp['status'] == 'pending':
        button_cols = container.columns([1, 1])
        with button_cols[0]:
            if st.button("🚀 Move Forward", key=f"forward_{rfp['id']}", use_container_width=True):
                stock_result = compare_rfp_with_stock(rfp['analysis'], STOCK_ITEMS)
                update_rfp_status(session_id, rfp['id'], 'move_forward', stock_match=stock_result)

                # build user-friendly popup message
                if stock_result.get('matches'):
                    lines = [f"✅ Stock match found for '{rfp['bid_title']}'"]
                    for match in stock_result['matches']:
                        item = match['matched_item']
                        enough = match['enough']
                        status_text = 'enough' if enough else 'not enough' if enough is False else 'availability unknown'
                        lines.append(
                            f"{match['product']} → {item.get('item_name', 'N/A')} ({item.get('range', 'N/A')}) requested {match.get('requested_qty', 'N/A')} / available {item.get('available', 'N/A')} ({status_text})"
                        )
                else:
                    lines = [f"⚠️ No stock match found for '{rfp['bid_title']}'"]

                if stock_result.get('no_matches'):
                    no_matches = ', '.join([p['product'] for p in stock_result['no_matches']])
                    lines.append(f"Products without stock match: {no_matches}")

                message_type = 'success' if stock_result.get('matches') and not stock_result.get('no_matches') else 'warning'
                st.session_state.last_stock_match = {
                    'type': message_type,
                    'message': '\n'.join(lines)
                }
                st.rerun()
        with button_cols[1]:
            if st.button("⏸️ Hold", key=f"hold_{rfp['id']}", use_container_width=True):
                update_rfp_status(session_id, rfp['id'], 'hold')
                st.rerun()


def show_rfp_cards(rfps, session_id, cards_per_row=3):
    for start in range(0, len(rfps), cards_per_row):
        row_rfps = rfps[start:start + cards_per_row]
        cols = st.columns(cards_per_row, gap='large')
        for col, rfp in zip(cols, row_rfps):
            display_rfp_card(rfp, session_id, parent=col)
def update_rfp_status(session_id, rfp_id, new_status, stock_match=None):
    """Update RFP status in session data"""
    session_file = os.path.join("sessions", f"session_{session_id}.json")

    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_data = json.load(f)

        for rfp in session_data['rfps']:
            if rfp['id'] == rfp_id:
                rfp['status'] = new_status
                if stock_match is not None:
                    rfp['stock_match'] = stock_match
                break

        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

def main():
    if 'last_stock_match' not in st.session_state:
        st.session_state.last_stock_match = None

    st.title("🤖 RFP Scraping Agent")
    st.markdown("Automated RFP discovery, extraction, and workflow management")
    inject_card_styles()

    if st.session_state.last_stock_match:
        payload = st.session_state.last_stock_match
        if payload.get('type') == 'success':
            st.success(payload.get('message'))
        else:
            st.warning(payload.get('message'))
        st.session_state.last_stock_match = None

    # Sidebar
    with st.sidebar:
        st.header("🎯 Agent Controls")

        if st.button("🚀 Start New Session", type="primary"):
            st.session_state.agent_running = True
            st.rerun()

        if st.button("⏹️ Stop Agent"):
            st.session_state.agent_running = False

        st.header("📊 Session Stats")
        sessions = load_sessions()

        if sessions:
            total_sessions = len(sessions)
            total_rfps = sum(len(s['rfps']) for s in sessions)
            move_forward = sum(len([r for r in s['rfps'] if r['status'] == 'move_forward']) for s in sessions)
            on_hold = sum(len([r for r in s['rfps'] if r['status'] == 'hold']) for s in sessions)

            st.metric("Total Sessions", total_sessions)
            st.metric("Total RFPs", total_rfps)
            st.metric("Move Forward", move_forward)
            st.metric("On Hold", on_hold)

    # Main content
    if st.session_state.agent_running:
        st.header("🔄 Agent Running")
        st.info("🤖 Agent is searching for RFPs and extracting information...")

        # Run the agent
        session_data = run_rfp_scraping_agent(keywords_limit=20, bids_per_keyword=2)

        # Save session
        save_session(session_data)
        st.session_state.agent_running = False

        st.success(f"✅ Session completed! Found {len(session_data['rfps'])} RFPs")
        st.rerun()

    # Display current session or select from history
    sessions = load_sessions()

    if not sessions:
        # If no sessions exist, try loading from catalog
        session_from_catalog = initialize_session_from_catalog()
        if session_from_catalog:
            sessions = [session_from_catalog]

    if sessions:
        st.header("📋 RFP Session Cards")

        # Session selector
        session_options = [f"{s['session_id']} ({len(s['rfps'])} RFPs)" for s in sessions]
        selected_session_idx = st.selectbox(
            "Select Session:",
            range(len(sessions)),
            format_func=lambda x: session_options[x]
        )

        selected_session = sessions[selected_session_idx]

        st.subheader(f"Session: {selected_session['session_id']}")
        st.caption(f"Started: {selected_session['start_time']}")

        # Display RFP cards - sorted by latest to oldest
        pending_rfps = sorted(
            [r for r in selected_session['rfps'] if r['status'] == 'pending'],
            key=lambda x: x['timestamp'],
            reverse=True
        )
        move_forward_rfps = sorted(
            [r for r in selected_session['rfps'] if r['status'] == 'move_forward'],
            key=lambda x: x['timestamp'],
            reverse=True
        )
        hold_rfps = sorted(
            [r for r in selected_session['rfps'] if r['status'] == 'hold'],
            key=lambda x: x['timestamp'],
            reverse=True
        )

        if pending_rfps:
            st.subheader("⏳ Pending RFPs")
            show_rfp_cards(pending_rfps, selected_session['session_id'])

        if move_forward_rfps:
            st.subheader("✅ Move Forward RFPs")
            show_rfp_cards(move_forward_rfps, selected_session['session_id'])

        if hold_rfps:
            st.subheader("⏸️ Hold RFPs")
            show_rfp_cards(hold_rfps, selected_session['session_id'])

    else:
        st.info("🤖 No sessions yet. Click 'Start New Session' to begin automated RFP discovery!")

if __name__ == "__main__":
    main()