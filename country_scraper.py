
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time

# Setup
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # optional
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Seite aufrufen
driver.get("https://taxsummaries.pwc.com/")
time.sleep(2)

# Klick auf das Dropdown-Menü ("Territory")
try:
    dropdown_toggle = driver.find_element(By.CLASS_NAME, "select2-selection")
    dropdown_toggle.click()
    time.sleep(2)

    # Alle Optionen in der geöffneten Liste extrahieren
    options_list = driver.find_elements(By.CSS_SELECTOR, "li.select2-results__option")

    # Alle Ländernamen auflisten
    countries = [opt.text for opt in options_list if opt.text.strip()]
    print("Länder gefunden:")
    for country in countries:
        print(country)

except Exception as e:
    print("Fehler:", e)

driver.quit()