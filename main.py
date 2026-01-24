import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from pyzbar.pyzbar import decode
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("data/phishing_dataset.csv")

df['content'] = df['content'].fillna("")
df['label'] = df['label'].astype(int)

print("Dataset loaded:", df.shape)

# ===============================
# 2. URL feature extraction
# ===============================
def extract_url_features(url):
    return [
        len(url),
        url.count('.'),
        url.count('-'),
        url.count('@'),
        url.count('?'),
        url.count('%'),
        url.count('/'),
        url.count('='),
        url.count('http'),
        url.count('https'),
        sum(c.isdigit() for c in url)
    ]

X_url = df['url'].apply(extract_url_features).tolist()
y = df['label']

scaler = StandardScaler()
X_url = scaler.fit_transform(X_url)

# ===============================
# 3. Content feature extraction
# ===============================
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_text = tfidf.fit_transform(df['content'])

# ===============================
# 4. Train models
# ===============================
X_url_train, X_url_test, y_train, y_test = train_test_split(
    X_url, y, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_url_train, y_train)

X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_text_train, y_text_train)

print("Models trained successfully")

# ===============================
# 5. Save models
# ===============================
import pickle
import os
os.makedirs("models", exist_ok=True)

with open("models/rf_url.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("models/lr_content.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("models/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ All models saved in 'models/' folder")


# ===============================
# 6. QR decode
# ===============================
def decode_qr(image_path):
    img = Image.open(image_path)
    result = decode(img)
    return result[0].data.decode() if result else None

# ===============================
# 7. Extract webpage content
# ===============================
def extract_web_content(url):
    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return " ".join(soup.get_text().split())
    except:
        return ""

# ===============================
# 8. Final prediction
# ===============================
def predict_phishing(input_value, is_qr=False):
    url = decode_qr(input_value) if is_qr else input_value
    if not url:
        return "Invalid input"

    url_feat = scaler.transform([extract_url_features(url)])
    url_prob = rf_model.predict_proba(url_feat)[0][1]

    content = extract_web_content(url)
    content_prob = 0.5
    if content.strip():
        content_vec = tfidf.transform([content])
        content_prob = lr_model.predict_proba(content_vec)[0][1]

    final_score = 0.5 * url_prob + 0.5 * content_prob
    return "⚠️ PHISHING" if final_score >= 0.5 else "✅ LEGITIMATE"

# ===============================
# 9. User interaction
# ===============================
print("\n1 → URL")
print("2 → QR Code")

choice = input("Choose input type: ")

if choice == "1":
    url = input("Enter URL: ")
    print(predict_phishing(url))
elif choice == "2":
    path = input("Enter QR image path: ")
    print(predict_phishing(path, is_qr=True))
else:
    print("Invalid choice")
