import streamlit as st
import pdfplumber
import requests
import json
import os
import re
from datetime import datetime
import tempfile

# Page configuration
st.set_page_config(
    page_title="RFP Information Extractor",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = []

# Functions
def extract_pdf_text(pdf_url, filename):
    """Extract text from PDF URL"""
    try:
        with st.spinner("Downloading PDF..."):
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name

        # Extract text
        text = ""
        with pdfplumber.open(temp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        # Clean up temp file
        os.unlink(temp_path)

        return text.strip()

    except Exception as e:
        st.error(f"Error extracting PDF: {e}")
        return None

def analyze_rfp_text(text):
    """Analyze RFP text and extract key information"""
    if not text:
        return {}

    analysis = {
        'full_text': text,
        'deadlines': [],
        'requirements': [],
        'contact_info': [],
        'financial_info': [],
        'bid_details': []
    }

    # Extract deadlines
    deadline_patterns = [
        r'deadline[:\s]*([^\n\r]+)',
        r'due date[:\s]*([^\n\r]+)',
        r'last date[:\s]*([^\n\r]+)',
        r'submission date[:\s]*([^\n\r]+)',
        r'bid submission[:\s]*([^\n\r]+)',
        r'tender closing[:\s]*([^\n\r]+)'
    ]

    for pattern in deadline_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        analysis['deadlines'].extend(matches)

    # Extract contact information
    contact_patterns = [
        r'contact[:\s]*([^\n\r]+)',
        r'email[:\s]*([^\n\r]+)',
        r'phone[:\s]*([^\n\r]+)',
        r'mobile[:\s]*([^\n\r]+)',
        r'tel[:\s]*([^\n\r]+)',
        r'contact person[:\s]*([^\n\r]+)'
    ]

    for pattern in contact_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        analysis['contact_info'].extend(matches)

    # Extract requirements
    requirement_keywords = [
        'requirement', 'specification', 'technical spec', 'scope of work',
        'eligibility', 'qualification', 'experience required', 'mandatory',
        'essential', 'criteria', 'certification', 'license'
    ]

    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in requirement_keywords):
            if len(line.strip()) > 10:
                analysis['requirements'].append(line.strip())

    # Extract financial information
    financial_patterns = [
        r'estimated cost[:\s]*([^\n\r]+)',
        r'budget[:\s]*([^\n\r]+)',
        r'value[:\s]*([^\n\r]+)',
        r'amount[:\s]*([^\n\r]+)',
        r'rs\.?\s*[\d,]+(?:\s*(?:lakh|crore|thousand|million)?)',
        r'inr\s*[\d,]+(?:\s*(?:lakh|crore|thousand|million)?)',
        r'₹\s*[\d,]+(?:\s*(?:lakh|crore|thousand|million)?)'
    ]

    for pattern in financial_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        analysis['financial_info'].extend(matches)

    # Extract bid details
    bid_patterns = [
        r'bid number[:\s]*([^\n\r]+)',
        r'tender number[:\s]*([^\n\r]+)',
        r'gem/([^\s]+)',
        r'tender id[:\s]*([^\n\r]+)',
        r'procurement type[:\s]*([^\n\r]+)'
    ]

    for pattern in bid_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        analysis['bid_details'].extend(matches)

    return analysis

def save_analysis(analysis, filename):
    """Save the analysis to a JSON file"""
    os.makedirs("extracted_data", exist_ok=True)

    analysis_data = {
        'filename': filename,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'analysis': analysis
    }

    safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
    analysis_file = os.path.join("extracted_data", f"{safe_filename.replace('.pdf', '')}_analysis.json")

    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)

    return analysis_file

# Main app
def main():
    st.title("📄 RFP Information Extractor")
    st.markdown("Extract key information from RFP PDF documents instead of just downloading them.")

    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This tool extracts and categorizes information from RFP PDFs:
        - 📅 Deadlines & submission dates
        - 📞 Contact information
        - 📋 Requirements & specifications
        - 💰 Financial information
        - 🏷️ Bid details
        """)

        st.header("📁 Saved Analyses")
        if os.path.exists("extracted_data"):
            analysis_files = [f for f in os.listdir("extracted_data") if f.endswith('_analysis.json')]
            if analysis_files:
                for file in analysis_files:
                    if st.button(f"📄 {file.replace('_analysis.json', '')}", key=file):
                        with open(os.path.join("extracted_data", file), 'r') as f:
                            data = json.load(f)
                            st.session_state.selected_analysis = data
            else:
                st.info("No saved analyses yet")

    # Main content
    tab1, tab2 = st.tabs(["🔍 Extract Information", "💾 Saved Analyses"])

    with tab1:
        st.header("Extract Information from RFP PDF")

        # Input form
        with st.form("extraction_form"):
            pdf_url = st.text_input(
                "PDF URL",
                placeholder="https://example.com/rfp-document.pdf",
                help="Enter the URL of the RFP PDF document"
            )

            filename = st.text_input(
                "Filename (optional)",
                placeholder="rfp_document.pdf",
                help="Optional filename for saving the analysis"
            )

            submitted = st.form_submit_button("🚀 Extract Information")

        if submitted and pdf_url:
            if not filename:
                filename = f"extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            # Extract and analyze
            extracted_text = extract_pdf_text(pdf_url, filename)

            if extracted_text:
                analysis = analyze_rfp_text(extracted_text)

                # Save analysis
                saved_file = save_analysis(analysis, filename)

                # Add to session state
                st.session_state.extracted_data.append({
                    'filename': filename,
                    'analysis': analysis,
                    'saved_file': saved_file
                })

                st.success(f"✅ Information extracted and saved to {saved_file}")

                # Display results
                display_analysis(analysis, filename)
            else:
                st.error("❌ Failed to extract text from PDF")

    with tab2:
        st.header("Saved Analyses")

        if 'selected_analysis' in st.session_state:
            data = st.session_state.selected_analysis
            st.subheader(f"📄 {data['filename']}")
            st.caption(f"Analyzed on: {data['timestamp']}")
            display_analysis(data['analysis'], data['filename'])
        else:
            st.info("Select an analysis from the sidebar to view details")

def display_analysis(analysis, filename):
    """Display the analysis results in a nice format"""

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Locations", len(analysis.get('location', [])))

    with col2:
        st.metric("Contacts", len(analysis.get('contacts', [])))

    with col3:
        st.metric("Products", len(analysis.get('products', [])))

    with col4:
        st.metric("Quantities", len(analysis.get('quantity', [])))

    with col5:
        st.metric("Bid Details", len(analysis.get('bid_details', [])))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Location")
        if analysis.get('location'):
            for location in analysis['location']:
                st.write(f"• {location}")
        else:
            st.write("*No location found*")

        st.subheader("📞 Contacts")
        if analysis.get('contacts'):
            for contact in analysis['contacts']:
                st.write(f"• {contact}")
        else:
            st.write("*No contacts found*")

        st.subheader("🏢 Organisation")
        if analysis.get('organisation_name'):
            for org in analysis['organisation_name']:
                st.write(f"• {org}")
        else:
            st.write("*No organisation name found*")

    with col2:
        st.subheader("📅 Bid Dates")
        if analysis.get('bid_opening_date'):
            st.write("**Opening:**")
            for d in analysis['bid_opening_date']:
                st.write(f"• {d}")
        else:
            st.write("*No opening date found*")

        if analysis.get('bid_end_date'):
            st.write("**Closing:**")
            for d in analysis['bid_end_date']:
                st.write(f"• {d}")
        else:
            st.write("*No closing date found*")

        st.subheader("💰 Financial")
        if analysis.get('financial_info'):
            for fin in analysis['financial_info']:
                st.write(f"• {fin}")
        else:
            st.write("*No financial info found*")

    st.subheader("📦 Products & Qty")
    if analysis.get('products') or analysis.get('quantity'):
        for product in analysis.get('products', []):
            st.write(f"• {product}")
        if analysis.get('quantity'):
            st.write("\nQuantities:")
            for qty in analysis.get('quantity', []):
                st.write(f"• {qty}")
    else:
        st.write("*No products or quantities found*")

    st.subheader("🏷️ Bid Details")
    if analysis.get('bid_details'):
        for bid in analysis['bid_details']:
            st.write(f"• {bid}")
    else:
        st.write("*No bid details found*")

    st.subheader("🧾 Evaluation Schedules")
    if analysis.get('evaluation_schedules'):
        for schedule in analysis['evaluation_schedules']:
            st.write(f"• {schedule.get('item')} — Qty: {schedule.get('quantity')}")
    else:
        st.write("*No evaluation schedules found*")

    # Full text (collapsible)
    with st.expander("📄 Full Extracted Text"):
        st.text_area(
            "Extracted text",
            analysis.get('full_text', ''),
            height=300,
            disabled=True
        )

if __name__ == "__main__":
    main()