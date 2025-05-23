from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import undetected_chromedriver as uc


# Global variable for buy percentage
BUY_PERCENTAGE = 0.15

# Initialize WebDriver
driver = uc.Chrome()
driver.get("https://www.pionex.us/")

driver.maximize_window()

# Wait for user to sign in
input("Log in to TradingView, open Pionex in the next tab, navigate to the trading panel, and then press Enter to continue...")

def buy():
    print("Placing Buy Order...")
    buy_button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, "//div[@role='tab' and text()='Buy']"))
        )
    buy_button.click()
    market_tab = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//div[@role='tab' and text()='Market']"))
    )
    market_tab.click()
    
    balance_span = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((
            By.XPATH, "//span[text()='Available balance:']/following-sibling::span"
        ))
    )

    # Extract and clean balance value
    balance_text = balance_span.text.strip().replace("USD", "").replace(",", "")
    balance_value = float(balance_text)

    # Calculate amount
    # amount = balance_value * BUY_PERCENTAGE
    amount = 20
    
    input_box = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//input[@placeholder[contains(., 'Min amount')]]"))
    )
    input_box.clear()
    input_box.send_keys(str(amount))
    
    buy_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Buy XRP']]"))
    )
    buy_button.click()

    print(f"Buy Order Placed: Bought ${amount} of coin")

def sell():
    print("Placing Sell Order...")
    sell_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//div[@role='tab' and text()='Sell']"))
    )
    sell_button.click()
    market_tab = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//div[@role='tab' and text()='Market']"))
    )
    market_tab.click()
    
    balance_span = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((
            By.XPATH, "//span[text()='Available balance:']/following-sibling::span[contains(text(), 'XRP')]"
        ))
    )

    # Extract and clean XRP balance
    balance_text = balance_span.text.strip()
    balance_number = balance_text.replace("XRP", "").replace(",", "").strip()

    balance_value = float(balance_number)
    
    input_box = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//input[@placeholder[contains(., 'Min amount')]]"))
    )
    input_box.clear()
    input_box.send_keys(str(balance_value))
    
    sell_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Sell XRP']]"))
    )
    sell_button.click()

    print(f"Sell Order Placed: Sold ${balance_value} of coin")

# while True:
#     message, timestamp = get_latest_alert()
#     if message and timestamp and (message != previous_message or timestamp != previous_timestamp):
#         print(f"New Alert: {message} at time {timestamp}")
#         if "order sell" in message:
#             sell()
#         elif "order buy" in message:
#             buy()
#         previous_message, previous_timestamp = message, timestamp
#     time.sleep(5)

if __name__ == "__main__":
    buy()
    time.sleep(5)
    sell()
    driver.quit()