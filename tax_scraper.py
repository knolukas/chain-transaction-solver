from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

def get_tax_rate(country):
    # Browser-Optionen (headless, damit kein Fenster aufgeht)
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Entferne diese Zeile, wenn du den Browser sehen willst

    # Starte den Browser
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # Gehe zur Seite
    url = f"https://taxsummaries.pwc.com/{country}"
    driver.get(url)
    time.sleep(3)  # Warte, bis die Seite vollständig geladen ist

    # Finde die Tabelle mit dem VAT-Wert
    try:
        vat_value = driver.find_element(
            By.XPATH,
            "//th[contains(text(),'Standard VAT rate')]/following-sibling::td/p"
        ).text
        print(f"Standard VAT rate for {country}: {vat_value}%")
        driver.quit()
        return int(vat_value)
    except Exception as e:
        print("Fehler beim Auslesen:", e)
        # Browser beenden
        driver.quit()
        return int(-1)




