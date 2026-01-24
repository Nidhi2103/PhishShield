# app.py
import streamlit as st
from PIL import Image
from pyzbar.pyzbar import decode
import requests
from bs4 import BeautifulSoup
import pickle
import os

st.set_page_config(page_title="PhishShield 🛡️", layout="centered")

# ---------------- Load trained models ----------------
try:
    with open("models/rf_url.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("models/lr_content.pkl", "rb") as f:
        lr_model = pickle.load(f)
    with open("models/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    st.error("❌ Models not found in 'models/' folder. Please train and save rf_url.pkl, lr_content.pkl, tfidf.pkl, and scaler.pkl.")
    st.stop()

# ---------------- Helper functions ----------------
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

@st.cache_data
def extract_web_content(url):
    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return " ".join(soup.get_text().split())
    except:
        return ""

def decode_qr(image):
    result = decode(Image.open(image))
    if result:
        return result[0].data.decode("utf-8")
    return None

def predict_phishing(url):
    # --- URL features ---
    url_feat = scaler.transform([extract_url_features(url)])
    url_prob = rf_model.predict_proba(url_feat)[0][1]

    # --- Web content ---
    content = extract_web_content(url)
    if content.strip() == "":
        content_prob = 0.5
    else:
        content_vec = tfidf.transform([content])
        content_prob = lr_model.predict_proba(content_vec)[0][1]

    # --- Ensemble ---
    final_score = 0.5 * url_prob + 0.5 * content_prob
    result = "⚠️ PHISHING" if final_score >= 0.5 else "✅ LEGITIMATE"
    return round(final_score, 2), result

# ---------------- Streamlit UI ----------------
st.title("🛡️ PhishShield: URL & QR Phishing Detection")
st.markdown("Enter a URL or upload a QR code image to check for phishing.")

option = st.radio("Choose input type:", ["URL", "QR Code Image"])

if option == "URL":
    url_input = st.text_input("Enter the URL here:")
    if st.button("Check URL") and url_input:
        prob, result = predict_phishing(url_input)
        color = "red" if "PHISHING" in result else "green"
        st.markdown(f"<h3 style='color:{color}'>{result} | Probability: {prob}</h3>", unsafe_allow_html=True)

else:
    qr_file = st.file_uploader("Upload QR Code image:", type=["png", "jpg", "jpeg"])
    if st.button("Check QR Code") and qr_file:
        decoded_url = decode_qr(qr_file)
        if decoded_url:
            st.info(f"Decoded URL: {decoded_url}")
            prob, result = predict_phishing(decoded_url)
            color = "red" if "PHISHING" in result else "green"
            st.markdown(f"<h3 style='color:{color}'>{result} | Probability: {prob}</h3>", unsafe_allow_html=True)
        else:
            st.error("❌ Could not decode QR code")
