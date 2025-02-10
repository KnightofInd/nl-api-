import requests

url = "https://nl-api.onrender.com/summarization/"
data = {
    "text": "Unit economics, a critical component of financial modeling...",  # your text here
    "max_length": 1000,
    "min_length": 50
}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raises an HTTPError for bad responses (4xx, 5xx)
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")