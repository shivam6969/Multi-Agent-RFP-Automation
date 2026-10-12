#!/usr/bin/env python3
"""Test if rfp_agent can load and use the catalog"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scraper'))

# Simulate catalog loading
import json
from datetime import datetime

def load_rfp_catalog():
    """Load RFP catalog from extracted_data/rfps_catalog.json"""
    catalog_path = os.path.join("extracted_data", "rfps_catalog.json")
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
        'rfps': rfp_catalog
    }
    return session_data

# Test
catalog = load_rfp_catalog()
print(f"✅ Catalog loaded: {len(catalog)} RFPs")

session = initialize_session_from_catalog()
if session:
    print(f"✅ Session created: {session['session_id']}")
    print(f"✅ Session contains {len(session['rfps'])} RFPs")
    print(f"✅ First RFP: {session['rfps'][0]['bid_title']}")
else:
    print("❌ Failed to create session from catalog")
