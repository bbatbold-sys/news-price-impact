"""
Investopedia Auto-Trader — Manual Login Version
Uses undetected-chromedriver + correct Vuetify selectors.

Rules:
  - Only trade if confidence > 90%
  - BUY signal  → always buy (even if already owned)
  - SELL signal + own stock → regular Sell (exit position)
  - SELL signal + don't own → Sell Short

Run: python investopedia_trader.py
"""
import sys, os, time, logging, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "web"))
os.chdir(os.path.join(os.path.dirname(__file__), "web"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("trader")

MIN_CONFIDENCE = 0.90
MAX_TRADES     = 5

def get_signals():
    from db import recent_news
    signals = recent_news(500)
    by_asset = {}
    for s in signals:
        a = s["asset"]
        if a not in by_asset:
            by_asset[a] = s
    buys, sells = [], []
    for asset, s in by_asset.items():
        if "=F" in asset or "-USD" in asset:
            continue
        conf = s.get("confidence_T3", 0) or 0
        dir3 = s.get("direction_T3", "FLAT")
        if conf >= MIN_CONFIDENCE:
            if dir3 == "UP":     buys.append((asset, conf))
            elif dir3 == "DOWN": sells.append((asset, conf))
    buys.sort(key=lambda x: x[1], reverse=True)
    sells.sort(key=lambda x: x[1], reverse=True)
    return buys, sells

def get_portfolio_holdings(driver, wait):
    """Return set of tickers currently held."""
    from selenium.webdriver.common.by import By
    held = set()
    try:
        driver.get("https://www.investopedia.com/simulator/portfolio")
        time.sleep(4)
        links = driver.find_elements(By.CSS_SELECTOR, "table td a")
        for el in links:
            txt = el.text.strip()
            if re.match(r'^[A-Z]{1,5}$', txt):
                held.add(txt)
    except Exception as e:
        log.warning(f"Could not fetch portfolio: {e}")
    log.info(f"Holdings: {held if held else 'none'}")
    return held

def click_dropdown_option(driver, option_text):
    """After opening a Vuetify dropdown, click the matching option."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    wait = WebDriverWait(driver, 8)
    # Vuetify renders options as .v-list-item or [role="option"]
    opts = wait.until(EC.presence_of_all_elements_located(
        (By.CSS_SELECTOR, ".v-list-item, [role='option']")))
    for opt in opts:
        if option_text.lower() in opt.text.lower():
            opt.click()
            return True
    log.warning(f"Option '{option_text}' not found. Available: {[o.text for o in opts]}")
    return False

def js_click(driver, element):
    """Click using JavaScript to bypass overlay interceptions."""
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)
    time.sleep(0.3)
    driver.execute_script("arguments[0].click();", element)

def place_trade(driver, ticker, action):
    """
    action: "BUY", "SELL" (own it), or "SHORT" (don't own it)
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys

    wait = WebDriverWait(driver, 15)
    action_text = {"BUY": "Buy", "SELL": "Sell", "SHORT": "Sell Short"}[action]
    label       = {"BUY": "BUY", "SELL": "SELL (exit)", "SHORT": "SELL SHORT"}[action]
    log.info(f"  Placing {label} for {ticker}...")

    try:
        driver.get("https://www.investopedia.com/simulator/trade/stocks")
        time.sleep(6)

        # ── 1. Type symbol ─────────────────────────────────────
        sym = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, '[data-cy="symbol-search-field"] input')))
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", sym)
        time.sleep(0.5)
        sym.click(); sym.clear()
        sym.send_keys(ticker)
        time.sleep(2)

        # ── 2. Click autocomplete suggestion ──────────────────
        # The autocomplete dropdown shows matching tickers — click the exact match
        suggestion = wait.until(EC.element_to_be_clickable(
            (By.XPATH, f"//div[contains(@class,'v-list-item')]//div[normalize-space()='{ticker}']/..")))
        js_click(driver, suggestion)
        time.sleep(1.5)

        # ── 3. Action dropdown ─────────────────────────────────
        # Dismiss cookie banner if still visible
        try:
            cb = driver.find_element(By.ID, "onetrust-accept-btn-handler")
            driver.execute_script("arguments[0].click();", cb)
            time.sleep(0.5)
        except Exception:
            pass

        action_field = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, '[data-cy="action-select"] [role="select-action"]')))
        js_click(driver, action_field)
        time.sleep(1.5)

        # Find dropdown options — try multiple selectors
        opts = driver.find_elements(By.CSS_SELECTOR,
            '[role="option"], .v-list-item--link, .v-overlay--active .v-list-item')
        # Filter out invisible ones
        opts = [o for o in opts if o.is_displayed() and o.text.strip()]
        log.info(f"  Action options found: {[o.text.strip() for o in opts]}")

        clicked = False
        for opt in opts:
            if action_text.lower() in opt.text.lower():
                js_click(driver, opt)
                clicked = True
                break
        if not clicked:
            raise Exception(f"Action '{action_text}' not found. Options: {[o.text for o in opts]}")
        time.sleep(0.8)

        # ── 4. Quantity ────────────────────────────────────────
        qty = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, '[data-cy="quantity-input"] input')))
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", qty)
        time.sleep(0.3)
        qty.click(); qty.clear()
        qty.send_keys("1")
        time.sleep(0.5)

        # ── 5. Preview ─────────────────────────────────────────
        preview = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, '[data-cy="preview-button"]')))
        js_click(driver, preview)
        time.sleep(4)

        # ── 6. Submit ──────────────────────────────────────────
        submit = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, '[data-cy="submit-button"], button[type="submit"]')))
        js_click(driver, submit)
        time.sleep(2)

        log.info(f"  ✓ {label} {ticker} submitted!")
        return True

    except Exception as e:
        try:
            driver.save_screenshot(f"C:/Users/batbo/news-price-impact/fail_{ticker}.png")
        except Exception:
            pass
        log.warning(f"  ✗ {label} {ticker} failed: {e}")
        return False

def run():
    import undetected_chromedriver as uc
    from selenium.webdriver.support.ui import WebDriverWait

    log.info("Getting signals...")
    buys, sells = get_signals()
    log.info(f"BUY  signals (>90%): {[f'{a}({c:.0%})' for a,c in buys[:5]]}")
    log.info(f"SELL signals (>90%): {[f'{a}({c:.0%})' for a,c in sells[:5]]}")

    if not buys and not sells:
        log.info("No signals above 90% confidence. Nothing to trade.")
        return

    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = uc.Chrome(options=options, version_main=145)
    wait   = WebDriverWait(driver, 20)

    driver.get("https://www.investopedia.com/simulator/")

    log.info("=" * 55)
    log.info(">>> BROWSER IS OPEN — PLEASE LOG IN MANUALLY <<<")
    log.info(">>> Navigate to your Portfolio page when done  <<<")
    log.info("=" * 55)

    logged_in = False
    for i in range(180):
        time.sleep(1)
        try:
            url = driver.current_url
        except Exception:
            log.warning("Browser closed."); return
        if "simulator/portfolio" in url or "simulator/home" in url:
            log.info(f"Logged in! URL: {url}")
            logged_in = True
            break
        if i % 15 == 0 and i > 0:
            log.info(f"  Still waiting... ({i}s) {url[:70]}")

    if not logged_in:
        log.warning("Timed out."); driver.quit(); return

    time.sleep(2)

    # Accept cookies banner (blocks clicks if visible)
    try:
        btn = driver.find_element(By.ID, "onetrust-accept-btn-handler")
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(1)
        log.info("Cookies accepted")
    except Exception:
        pass

    log.info("Checking portfolio holdings...")
    held = get_portfolio_holdings(driver, wait)

    log.info("Starting trades...")
    done = 0

    for ticker, conf in buys[:MAX_TRADES]:
        log.info(f"BUY {ticker} ({conf:.0%})")
        if place_trade(driver, ticker, "BUY"):
            done += 1
        time.sleep(1)

    for ticker, conf in sells[:MAX_TRADES]:
        if ticker in held:
            log.info(f"SELL {ticker} ({conf:.0%}) — owned")
            action = "SELL"
        else:
            log.info(f"SELL SHORT {ticker} ({conf:.0%}) — not owned")
            action = "SHORT"
        if place_trade(driver, ticker, action):
            done += 1
        time.sleep(1)

    log.info(f"All done — {done} trade(s) placed.")
    log.info("Browser stays open 30s for review.")
    time.sleep(30)
    driver.quit()

if __name__ == "__main__":
    run()
