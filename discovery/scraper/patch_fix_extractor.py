from pathlib import Path

path = Path('c:/Users/vidip/OneDrive/Desktop/PRJ3/scraper/rfp_extractor.py')
text = path.read_text(encoding='utf-8')
start = text.index('def analyze_rfp_text(text):')
end = text.index('def save_analysis', start)
new_body = '''def analyze_rfp_text(text):
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
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        for i, line in enumerate(lines):
            qty_match = re.search(r'\b(?:qty|quantity|units?|nos?|pieces?)[:\s]*([0-9,]+)\b', line, re.IGNORECASE)
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
        r'start[\\s]+time[:\\s]*([^\\n\\r]+)',
        r'opening[\\s]+time[:\\s]*([^\\n\\r]+)',
        r'bid[\\s]+opening[\\s]+time[:\\s]*([^\\n\\r]+)',
        r'from[:\\s]*([0-9]{1,2}:[0-9]{2}\\s*(?:am|pm|AM|PM)?)',
        r'starts?[:\\s]+([0-9]{1,2}:[0-9]{2}\\s*(?:am|pm|AM|PM)?)'
    ], text)

    analysis['end_time'] = _find_matches([
        r'end[\\s]+time[:\\s]*([^\\n\\r]+)',
        r'closing[\\s]+time[:\\s]*([^\\n\\r]+)',
        r'bid[\\s]+closing[\\s]+time[:\\s]*([^\\n\\r]+)',
        r'till[\\s]+([0-9]{1,2}:[0-9]{2}\\s*(?:am|pm|AM|PM)?)',
        r'to[:\\s]*([0-9]{1,2}:[0-9]{2}\\s*(?:am|pm|AM|PM)?)'
    ], text)

    analysis['location'] = _find_matches([
        r'location[:\\s]*([^\\n\\r]+)',
        r'place[:\\s]*([^\\n\\r]+)',
        r'venue[:\\s]*([^\\n\\r]+)',
        r'address[:\\s]*([^\\n\\r]+)',
        r'office[\\s]+at[:\\s]*([^\\n\\r]+)',
        r'located[\\s]+at[:\\s]*([^\\n\\r]+)'
    ], text)

    analysis['contacts'] = _find_matches([
        r'(?:contact(?: person| details)?|contact name|contact us|procurement officer|issuer)[:\\s]*([^\\n\\r]+)',
        r'email[:\\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})',
        r'phone[:\\s]*([\\d+\\-()\\\\s]{7,})',
        r'mobile[:\\s]*([\\d+\\-()\\\\s]{7,})',
        r'tel[:\\s]*([\\d+\\-()\\\\s]{7,})'
    ], text)

    analysis['products'], analysis['quantity'] = _extract_products_and_quantities(text)

    analysis['financial_info'] = _find_matches([
        r'estimated cost[:\\s]*([^\\n\\r]+)',
        r'budget[:\\s]*([^\\n\\r]+)',
        r'value[:\\s]*([^\\n\\r]+)',
        r'amount[:\\s]*([^\\n\\r]+)',
        r'price[:\\s]*([^\\n\\r]+)',
        r'cost[:\\s]*([^\\n\\r]+)',
        r'rs\\.?\\s*[\\d,]+(?:\\s*(?:lakh|crore|thousand|million)?)',
        r'inr\\s*[\\d,]+(?:\\s*(?:lakh|crore|thousand|million)?)',
        r'₹\\s*[\\d,]+(?:\\s*(?:lakh|crore|thousand|million)?)'
    ], text)

    analysis['bid_details'] = _find_matches([
        r'\\b(?:tender number|bid number|tender no|bid no|tender id|rfp number)[:\\s]*([^\\n\\r]+)',
        r'procurement type[:\\s]*([^\\n\\r]+)',
        r'procurement mode[:\\s]*([^\\n\\r]+)'
    ], text)

    for match in re.findall(r'\\bGEM/\\d{4}/[A-Z]+/\\d{4,}\\b', text, re.IGNORECASE):
        if match not in analysis['bid_details']:
            analysis['bid_details'].append(match)

    analysis['bid_end_date'] = _find_matches([
        r'bid end date[:\\s]*([^\\n\\r]+)',
        r'bid closing date[:\\s]*([^\\n\\r]+)',
        r'tender closing date[:\\s]*([^\\n\\r]+)',
        r'last date of submission[:\\s]*([^\\n\\r]+)',
        r'deadline[:\\s]*([^\\n\\r]+)',
        r'due date[:\\s]*([^\\n\\r]+)'
    ], text)

    analysis['bid_opening_date'] = _find_matches([
        r'bid opening date[:\\s]*([^\\n\\r]+)',
        r'tender opening date[:\\s]*([^\\n\\r]+)',
        r'date of opening[:\\s]*([^\\n\\r]+)',
        r'opening date[:\\s]*([^\\n\\r]+)'
    ], text)

    analysis['organisation_name'] = _find_matches([
        r'organisation[:\\s]*([^\\n\\r]+)',
        r'organization[:\\s]*([^\\n\\r]+)',
        r'issued by[:\\s]*([^\\n\\r]+)',
        r'published by[:\\s]*([^\\n\\r]+)',
        r'department[:\\s]*([^\\n\\r]+)',
        r'ministry[:\\s]*([^\\n\\r]+)'
    ], text)

    schedule_section_match = re.search(
        r'evaluation schedules?|schedule.*items?|items?.*schedule.*quantity|item.*category.*quantity',
        text, re.IGNORECASE
    )
    if schedule_section_match:
        schedule_start = schedule_section_match.start()
        schedule_text = text[schedule_start:schedule_start + 2000]
        schedule_lines = [line.strip() for line in schedule_text.split('\\n') if line.strip()]
        for line in schedule_lines:
            if re.search(r'schedule\\s+\\d+|item\\s+\\d+', line, re.IGNORECASE):
                qty_match = re.search(r'([0-9,]+)(?:\\s*$|\\s*[^\\w])', line)
                if qty_match:
                    item_desc = re.sub(r'schedule\\s+\\d+|item\\s+\\d+|[0-9,]+\\s*$', '', line, flags=re.IGNORECASE).strip()
                    if item_desc and len(item_desc) > 5:
                        analysis['evaluation_schedules'].append({
                            'item': item_desc,
                            'quantity': qty_match.group(1).replace(',', '').strip()
                        })

    return analysis
'''
text = text[:start] + new_body + text[end:]
path.write_text(text, encoding='utf-8')
print('Updated rfp_extractor.py analyzer block')
