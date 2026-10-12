import pdfplumber
import re
import tempfile
import requests

# -------------------------------
# TEXT EXTRACTION
# -------------------------------
def extract_text_from_pdf(pdf_url):
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(response.content)
            temp_path = temp.name

        text = ""
        tables = []

        with pdfplumber.open(temp_path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)

        return text, tables

    except Exception as e:
        print("PDF Extraction Error:", e)
        return None, []


# -------------------------------
# TABLE EXTRACTION
# -------------------------------
def extract_items_from_tables(tables):
    items = []

    for table in tables:
        for row in table:
            if not row:
                continue

            row = [str(cell).strip() if cell else "" for cell in row]

            if len(row) >= 3:
                item = row[1]
                qty = row[-1]

                qty_clean = re.sub(r'\D', '', qty)

                if item and qty_clean:
                    items.append({
                        "item": item,
                        "quantity": int(qty_clean)
                    })

    return items


# -------------------------------
# TEXT ANALYSIS
# -------------------------------
def analyze_text(text):
    data = {
        "bid_end_date": None,
        "bid_opening_date": None,
        "organization_name": None
    }

    match = re.search(r'(bid end date|closing date)[:\s]*([^\n\r]+)', text, re.I)
    if match:
        data["bid_end_date"] = match.group(2).strip()

    match = re.search(r'(bid opening date)[:\s]*([^\n\r]+)', text, re.I)
    if match:
        data["bid_opening_date"] = match.group(2).strip()

    match = re.search(r'(organisation|organization|ministry|department)[:\s]*([^\n\r]+)', text, re.I)
    if match:
        data["organization_name"] = match.group(2).strip()

    return data