# 🛡️ PhishShield

PhishShield is a Machine Learning–based phishing and QR code detection system with a Streamlit interface.

## Features
- URL phishing detection
- QR code phishing detection
- Random Forest (URL features)
- Logistic Regression + TF-IDF (content features)
- Streamlit web interface

## Tech Stack
- Python
- Scikit-learn
- Streamlit
- BeautifulSoup
- Pyzbar

## How to Run
```bash
pip install -r requirements.txt
python main.py
streamlit run app.py