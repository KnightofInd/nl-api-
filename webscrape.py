import requests
from bs4 import BeautifulSoup

# Function to search Google for web pages and extract links and snippets
def search_google(topic, max_results=10):
    search_url = f"https://www.google.com/search?q={topic.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    for g in soup.find_all("div", class_="tF2Cxc"):
        link = g.find("a")["href"]
        snippet = g.find("span", class_="aCOpRe").text if g.find("span", class_="aCOpRe") else "No snippet available."
        results.append((link, snippet))

        if len(results) >= max_results:
            break
    return results

# Main function to scrape data based on user input
def main():
    topic = input("Enter the topic you want to search for: ")

    print(f"\nSearching for '{topic}' on Google...")
    google_results = search_google(topic)
    print("\nWeb Page Links and Snippets:")
    for idx, (link, snippet) in enumerate(google_results, start=1):
        print(f"{idx}. {link}\n   Snippet: {snippet}\n")

if __name__ == "__main__":
    main()
import requests
from bs4 import BeautifulSoup

# Function to search Google for web pages and extract links and snippets
def search_google(topic, max_results=10):
    search_url = f"https://www.google.com/search?q={topic.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    params = {
        "q": topic.replace(' ', '+'),
        "num": max_results
    }
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    for g in soup.find_all("div", class_="yuRUbf"):
        link = g.find("a")["href"]
        snippet = g.find("span", class_="IsZvec").text if g.find("span", class_="IsZvec") else "No snippet available."
        results.append((link, snippet))

    return results

# Main function to scrape data based on user input
def main():
    topic = input("Enter the topic you want to search for: ")

    print(f"\nSearching for '{topic}' on Google...")
    try:
        google_results = search_google(topic)
        print("\nWeb Page Links and Snippets:")
        for idx, (link, snippet) in enumerate(google_results, start=1):
            print(f"{idx}. {link}\n   Snippet: {snippet}\n")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()