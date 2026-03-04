"""
Saves the full HTML of the trade page so we can find element IDs.
"""
import time, logging
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

options = uc.ChromeOptions()
options.add_argument("--start-maximized")
driver = uc.Chrome(options=options, version_main=145)

driver.get("https://www.investopedia.com/simulator/")
log.info("Log in, go to portfolio page (3 min)...")
for i in range(180):
    time.sleep(1)
    try:
        url = driver.current_url
    except Exception:
        log.warning("Browser closed"); driver.quit(); exit()
    if "simulator/portfolio" in url or "simulator/home" in url:
        log.info("Portfolio detected!")
        break
    if i % 15 == 0 and i > 0:
        log.info(f"  Waiting... ({i}s)")
else:
    log.warning("Timed out"); driver.quit(); exit()

time.sleep(2)

# Accept cookies
try:
    driver.find_element(By.ID, "onetrust-accept-btn-handler").click()
    time.sleep(1)
    log.info("Cookies accepted")
except Exception:
    pass

# Go to stocks trade page
log.info("Going to trade/stocks...")
driver.get("https://www.investopedia.com/simulator/trade/stocks")
log.info("Waiting 20s for React to render...")
time.sleep(20)

driver.save_screenshot("C:/Users/batbo/news-price-impact/trade_loaded.png")
log.info("Screenshot saved: trade_loaded.png")

# Save full page HTML
html = driver.page_source
with open("C:/Users/batbo/news-price-impact/trade_page.html", "w", encoding="utf-8") as f:
    f.write(html)
log.info(f"HTML saved: trade_page.html ({len(html)} bytes)")

# Dump all interactive elements
log.info("--- Interactive elements ---")
for e in driver.find_elements(By.CSS_SELECTOR, "input, select, button, textarea"):
    log.info(f"  <{e.tag_name}> id='{e.get_attribute('id')}' "
             f"name='{e.get_attribute('name')}' "
             f"placeholder='{e.get_attribute('placeholder')}' "
             f"data-testid='{e.get_attribute('data-testid')}' "
             f"text='{e.text.strip()[:30]}'")

driver.quit()
log.info("Done.")
