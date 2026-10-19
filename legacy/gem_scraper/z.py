from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time

# -------------------------------
# KEYWORDS
# -------------------------------
ELECTRICAL_KEYWORDS = {
    
}

# Flatten all keywords
ALL_KEYWORDS = [kw for kws in ELECTRICAL_KEYWORDS.values() for kw in kws]

# -------------------------------
# CHROME DOWNLOAD SETUP
# -------------------------------
options = Options()
options.add_experimental_option("prefs", {
    "download.default_directory": r"C:\Users\vidip\Downloads\gem_pdfs",  #change to your desired pat
    "download.prompt_for_download": False,
    "plugins.always_open_pdf_externally": True
})

driver = webdriver.Chrome(options=options)
driver.get("https://bidplus.gem.gov.in/all-bids")

wait = WebDriverWait(driver, 20)

# -------------------------------
# MAIN LOOP
# -------------------------------
for keyword in ALL_KEYWORDS:
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

        # Download first 3 for each keyword
        for i in range(min(3, len(bids))):
            try:
                print("⬇️ Downloading:", bids[i].text)

                driver.execute_script("arguments[0].scrollIntoView();", bids[i])
                driver.execute_script("arguments[0].click();", bids[i])

                time.sleep(4)

            except Exception as e:
                print("❌ Download error:", e)

    except Exception as e:
        print("❌ Search failed:", e)

    # small delay before next keyword
    time.sleep(3)

# -------------------------------
# DONE
# -------------------------------
driver.quit()
print("\n✅ All keywords processed")