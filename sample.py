import requests
url = "http://127.0.0.1:8000/sentiment-analysis/"
data = {"text": "FastAPI is amazing!"}

response = requests.post(url, json=data)
print(response.json())