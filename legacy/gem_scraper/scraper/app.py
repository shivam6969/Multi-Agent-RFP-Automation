import streamlit as st
import json
import os
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# 🔹 IMPORT YOUR EXTRACTOR
from extractor import extract_text_from_pdf, extract_items_from_tables, analyze_text

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(page_title="RFP Agent", layout="wide")

# -------------------------------
# KEYWORDS
# -------------------------------
KEYWORDS = [
    "cable", "transformer", "switch", "led", "electrical", "wire"
]

# -------------------------------
# PDF PROCESSING
# -------------------------------
def process_pdf(pdf_url):
    text, tables = extract_text_from_pdf(pdf_url)

    if not text:
        return None

    text_data = analyze_text(text)
    items = extract_items_from_tables(tables)

    return {
        "bid_end_date": text_data["bid_end_date"],
        "bid_opening_date": text_data["bid_opening_date"],
        "organisation_name": text_data["organization_name"],
        "evaluation_schedules": items
    }

# -------------------------------
# SCRAPER (IMPROVED)
# -------------------------------
def scrape_rfps():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 15)

    driver.get("https://bidplus.gem.gov.in/all-bids")

    all_rfps = []

    try:
        for keyword in KEYWORDS:
            st.write(f"🔍 Searching: {keyword}")

            search_box = wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Enter Keyword']"))
            )

            search_box.clear()
            search_box.send_keys(keyword)
            search_box.send_keys(Keys.RETURN)

            time.sleep(3)

            # 🔹 SCROLL TO LOAD MORE
            for _ in range(3):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)

            # 🔹 GET ALL PDF LINKS
            links = driver.find_elements(By.XPATH, "//a[contains(@href,'.pdf')]")

            st.write(f"Found {len(links)} PDFs")

            for link in links[:8]:  # increase limit
                pdf_url = link.get_attribute("href")
                title = link.text.strip() or "GEM BID"

                if pdf_url:
                    st.write(f"📄 Processing: {title}")

                    try:
                        analysis = process_pdf(pdf_url)

                        if analysis:
                            all_rfps.append({
                                "title": title,
                                "pdf_url": pdf_url,
                                "analysis": analysis,
                                "timestamp": datetime.now().isoformat()
                            })

                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")

    finally:
        driver.quit()

    return all_rfps

# -------------------------------
# SAVE SESSION
# -------------------------------
def save_data(data):
    os.makedirs("sessions", exist_ok=True)
    filename = f"sessions/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    return filename

# -------------------------------
# UI DISPLAY
# -------------------------------
def display_rfp(rfp):
    st.markdown("### 📌 " + rfp["title"])

    analysis = rfp["analysis"]

    col1, col2, col3 = st.columns(3)

    col1.metric("🏢 Organisation", analysis["organisation_name"] or "N/A")
    col2.metric("📅 End Date", analysis["bid_end_date"] or "N/A")
    col3.metric("📂 Opening Date", analysis["bid_opening_date"] or "N/A")

    # 🔹 TABLE DISPLAY
    schedules = analysis["evaluation_schedules"]

    if schedules:
        st.write("📋 **Items & Quantity**")
        st.table(schedules)

    st.link_button("🔗 Open PDF", rfp["pdf_url"])
    st.divider()

# -------------------------------
# MAIN APP
# -------------------------------
def main():
    st.title("🤖 RFP Scraping Agent (Improved)")

    if st.button("🚀 Start Scraping"):
        with st.spinner("Scraping RFPs..."):
            rfps = scrape_rfps()

        if rfps:
            save_data(rfps)
            st.success(f"✅ Extracted {len(rfps)} RFPs")

            for rfp in rfps:
                display_rfp(rfp)
        else:
            st.warning("No RFPs found")

if __name__ == "__main__":
    main()