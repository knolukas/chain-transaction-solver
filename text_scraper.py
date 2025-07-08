import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL der Zielseite
url = "https://linda.lindeverlag.at/Dokument/98485/"

# Header setzen, um wie ein Browser zu wirken
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# HTTP-Anfrage senden
response = requests.get(url, headers=headers)
response.raise_for_status()  # Fehler auslösen, wenn was schiefläuft

# HTML parsen
soup = BeautifulSoup(response.text, "html.parser")

# Ziel-Elemente finden
content_divs = soup.find_all("div", class_="hinweis sonstig content-box")

# Daten extrahieren
data = []
for div in content_divs:
    title = div.find("span", class_="title")
    paragraphs = div.find_all("p")

    title_text = title.get_text(strip=True) if title else ""
    paragraph_text = "\n\n".join(p.get_text(strip=True) for p in paragraphs)

    data.append({
        "Titel": title_text,
        "Inhalt": paragraph_text
    })

# In DataFrame umwandeln
df = pd.DataFrame(data)

# Optional: als CSV speichern
df.to_csv("scraped_lindeverlag.csv", index=False, encoding="utf-8-sig")

print(df)
