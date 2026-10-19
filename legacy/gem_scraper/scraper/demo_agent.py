#!/usr/bin/env python3
"""
Demo of RFP Agent Session Cards - UPDATED with compact 200px cards
Shows start/end time, location, contacts, products, quantity, and financial info
"""

import streamlit as st
import json
import os
from datetime import datetime

# Sample RFP data for demonstration with new fields
SAMPLE_RFPS = [
    {
        'id': 'demo_001',
        'keyword': 'lt cable',
        'bid_title': 'GEM/2024/B/123456 - Supply of LT Cables',
        'pdf_url': 'https://example.com/rfp1.pdf',
        'filename': 'GEM_2024_B_123456.pdf',
        'status': 'pending',
        'timestamp': datetime.now().isoformat(),
        'analysis': {
            'start_time': ['10:30 AM'],
            'end_time': ['03:30 PM'],
            'location': ['Head Office, New Delhi'],
            'contacts': ['Mr. Rajesh Kumar - 9876543210', 'email@gem.gov.in'],
            'products': ['LT Cable - Aluminium, 2.5 SQ MM, 90M Length'],
            'quantity': ['5000'],
            'financial_info': ['Estimated Cost: Rs. 25,00,000/-', 'Budget: 25 lakh'],
            'bid_details': ['GEM/2024/B/123456', 'Procurement Type: Buy']
        }
    },
    {
        'id': 'demo_002',
        'keyword': 'transformer',
        'bid_title': 'GEM/2024/B/234567 - Distribution Transformers',
        'pdf_url': 'https://example.com/rfp2.pdf',
        'filename': 'GEM_2024_B_234567.pdf',
        'status': 'pending',
        'timestamp': datetime.now().isoformat(),
        'analysis': {
            'start_time': ['09:00 AM'],
            'end_time': ['02:00 PM'],
            'location': ['Regional Office, Mumbai'],
            'contacts': ['Ms. Priya Singh - 8765432109', 'procurement@company.com'],
            'products': ['Distribution Transformer - 100 KVA, Oil Cooled'],
            'quantity': ['150'],
            'financial_info': ['Budget: Rs. 50,00,000/-', 'Estimated value: 50 lakh'],
            'bid_details': ['GEM/2024/B/234567', 'Category: Electrical']
        }
    },
    {
        'id': 'demo_003',
        'keyword': 'switchgear',
        'bid_title': 'GEM/2024/B/345678 - MV Switchgear Supply',
        'pdf_url': 'https://example.com/rfp3.pdf',
        'filename': 'GEM_2024_B_345678.pdf',
        'status': 'pending',
        'timestamp': datetime.now().isoformat(),
        'analysis': {
            'start_time': ['08:30 AM'],
            'end_time': ['04:30 PM'],
            'location': ['Bangalore Tech Park, Bangalore'],
            'contacts': ['Mr. Vikram Patel - 7654321098', 'tender@grid.com'],
            'products': ['MV Switchgear - 33 KV, Vacuum Type'],
            'quantity': ['85'],
            'financial_info': ['Budget: Rs. 75,00,000/-', 'Estimated: 75 lakh rupees'],
            'bid_details': ['GEM/2024/B/345678', 'Type: Procurement']
        }
    },
    {
        'id': 'demo_004',
        'keyword': 'motor',
        'bid_title': 'GEM/2024/B/456789 - Industrial Motors',
        'pdf_url': 'https://example.com/rfp4.pdf',
        'filename': 'GEM_2024_B_456789.pdf',
        'status': 'pending',
        'timestamp': datetime.now().isoformat(),
        'analysis': {
            'start_time': ['11:00 AM'],
            'end_time': ['05:00 PM'],
            'location': ['Chennai Industrial Zone, Chennai'],
            'contacts': ['Mr. Ashok Kumar - 6543210987', 'bids@motors.com'],
            'products': ['3-Phase Motor - 10 HP, 1440 RPM'],
            'quantity': ['200'],
            'financial_info': ['Budget: Rs. 35,00,000/-', 'Amount: 35 lakh'],
            'bid_details': ['GEM/2024/B/456789', 'Category: Motors']
        }
    }
]

def display_rfp_card(rfp):
    """Display an RFP as a clean card using Streamlit components"""
    # Prepare data
    start_time = rfp['analysis']['start_time'][0] if rfp['analysis']['start_time'] else 'N/A'
    end_time = rfp['analysis']['end_time'][0] if rfp['analysis']['end_time'] else 'N/A'
    location = rfp['analysis']['location'][0] if rfp['analysis']['location'] else 'N/A'
    contact = rfp['analysis']['contacts'][0] if rfp['analysis']['contacts'] else 'N/A'
    product = rfp['analysis']['products'][0] if rfp['analysis']['products'] else 'N/A'
    quantity = rfp['analysis']['quantity'][0] if rfp['analysis']['quantity'] else 'N/A'
    financial = rfp['analysis']['financial_info'][0] if rfp['analysis']['financial_info'] else 'N/A'
    
    # Determine status badge
    if rfp['status'] == 'pending':
        status_badge = '🟨 PENDING'
    elif rfp['status'] == 'move_forward':
        status_badge = '✅ MOVE FORWARD'
    else:
        status_badge = '🔴 ON HOLD'
    
    # Create card with Streamlit components
    with st.container(border=True):
        # Header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{rfp['bid_title'][:45]}**")
        with col2:
            st.markdown(f"*APR 2024*")
        
        st.divider()
        
        # Content
        st.markdown(f"**⏱️ Time:** {start_time} → {end_time}")
        st.markdown(f"**📍 Location:** {location}")
        st.markdown(f"**📞 Contact:** {contact[:40]}")
        st.markdown(f"**📦 Product:** {product[:45]}")
        st.markdown(f"**📊 Quantity:** {quantity} units")
        st.markdown(f"**💰 Budget:** {financial}")
        
        st.divider()
        
        # Status
        st.markdown(f"**Status:** {status_badge}")

def main():
    st.set_page_config(page_title="RFP Agent Demo", page_icon="🤖", layout="wide")
    st.title("🤖 RFP Agent Demo - Compact Session Cards")
    st.markdown("Interactive cards showing RFP information (200px width) with all key details")

    st.info("""
    ✨ **New Compact Card Features:**
    - **⏱️ Time**: Start and End time
    - **📍 Location**: Place of opening/submission
    - **📞 Contact**: Contact person details
    - **📦 Product**: Items needed
    - **📊 Quantity**: Number of units required
    - **💰 Financial**: Budget and cost information
    """)

    # Sidebar stats
    with st.sidebar:
        st.header("📊 Session Stats")

        total_rfps = len(SAMPLE_RFPS)
        move_forward = len([r for r in SAMPLE_RFPS if r['status'] == 'move_forward'])
        on_hold = len([r for r in SAMPLE_RFPS if r['status'] == 'hold'])
        pending = len([r for r in SAMPLE_RFPS if r['status'] == 'pending'])

        st.metric("Total RFPs", total_rfps)
        st.metric("Move Forward", move_forward)
        st.metric("On Hold", on_hold)
        st.metric("Pending", pending)

        st.markdown("---")

        if st.button("🔄 Reset All"):
            for rfp in SAMPLE_RFPS:
                rfp['status'] = 'pending'
            st.rerun()

    # Main content
    st.header("📋 RFP Session Cards")

    # Display RFP cards grouped by status - sorted by latest to oldest
    pending_rfps = sorted(
        [r for r in SAMPLE_RFPS if r['status'] == 'pending'],
        key=lambda x: x['timestamp'],
        reverse=True
    )
    move_forward_rfps = sorted(
        [r for r in SAMPLE_RFPS if r['status'] == 'move_forward'],
        key=lambda x: x['timestamp'],
        reverse=True
    )
    hold_rfps = sorted(
        [r for r in SAMPLE_RFPS if r['status'] == 'hold'],
        key=lambda x: x['timestamp'],
        reverse=True
    )

    if pending_rfps:
        st.subheader("⏳ Pending RFPs")
        cols = st.columns(3)
        for idx, rfp in enumerate(pending_rfps):
            col_idx = idx % 3
            with cols[col_idx]:
                display_rfp_card(rfp)
                if st.button("🚀 Forward", key=f"forward_{rfp['id']}", use_container_width=True):
                    rfp['status'] = 'move_forward'
                    st.rerun()
                if st.button("⏸️ Hold", key=f"hold_{rfp['id']}", use_container_width=True):
                    rfp['status'] = 'hold'
                    st.rerun()

    if move_forward_rfps:
        st.subheader("✅ Move Forward RFPs")
        st.success("These RFPs are ready for next workflow steps!")
        cols = st.columns(3)
        for idx, rfp in enumerate(move_forward_rfps):
            col_idx = idx % 3
            with cols[col_idx]:
                display_rfp_card(rfp)

    if hold_rfps:
        st.subheader("⏸️ Hold RFPs")
        st.warning("These RFPs are on hold for later review")
        cols = st.columns(3)
        for idx, rfp in enumerate(hold_rfps):
            col_idx = idx % 3
            with cols[col_idx]:
                display_rfp_card(rfp)

    # Instructions
    with st.expander("📖 Card Design & Features", expanded=False):
        st.markdown("""
        ### 📏 Card Specifications
        - **Width**: 200px (compact)
        - **Height**: 240px (minimum)
        - **Layout**: Flex wrap - multiple cards per row
        - **Responsive**: Adapts to screen size
        
        ### 🎯 Information Displayed
        1. **⏱️ Time Range**: Opening/Closing time
        2. **📍 Location**: Venue or submission place
        3. **📞 Contact**: Person to reach out to
        4. **📦 Product**: What items are needed
        5. **📊 Quantity**: Number of units
        6. **💰 Financial**: Budget information
        7. **Status**: Current workflow status
        
        ### 🚀 Actions
        - **Move Forward**: Proceed to next step
        - **Hold**: Review later
        - **Status Tracking**: Color-coded indicators
        """)

if __name__ == "__main__":
    main()