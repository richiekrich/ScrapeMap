import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from typing import Dict, List
import csv
import os

# Define patterns to search for using refined regex patterns
PATTERNS = {
    "Emails": r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b",
    "Credit Cards": r"\b(?:\d{4}[- ]?){3}\d{4}\b",
    "SSNs": r"\b\d{3}-\d{2}-\d{4}\b"
}

def analyze_text(data: str):
    findings = {key: [] for key in PATTERNS}
    for category, pattern in PATTERNS.items():
        matches = re.findall(pattern, data)
        if matches:
            findings[category].extend(matches)
    return findings

async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    """
    Asynchronously fetch website content.
    """
    try:
        async with session.get(url, raise_for_status=True) as response:
            return await response.text()
    except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError) as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

async def scrape_websites(urls: List[str]) -> Dict[str, str]:
    """
    Scrape multiple websites asynchronously and handle real data.
    """
    results = {}
    async with aiohttp.ClientSession(headers={"User-Agent": "Your Custom User-Agent Here"}) as session:
        tasks = [fetch(session, url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for url, response in zip(urls, responses):
            if isinstance(response, Exception):
                print(f"Error fetching {url}: {response}")
            else:
                results[url] = response
    return results


def create_dataset(data: Dict[str, str]) -> (List[List[int]], List[str]):
    """
    Create a dataset from the scraped data.
    """
    vectorizer = CountVectorizer()
    if data:
        X = vectorizer.fit_transform(data.values())
        return X, vectorizer.get_feature_names_out()
    return None, []

def train_model(X, y) -> MultinomialNB:
    """
    Train a machine learning model on the dataset.
    """
    model = MultinomialNB()
    model.fit(X, y)
    return model


def analyze_file(filepath: str) -> Dict[str, List[str]]:
    """
    Analyze a file for sensitive information based on predefined patterns. Handles both text and CSV files.
    """
    extension = os.path.splitext(filepath)[1].lower()
    print(f"Detected file extension: {extension}")  # Debug output
    data = ''

    try:
        if extension == '.txt':
            with open(filepath, 'r', encoding='utf-8') as file:
                data = file.read()
                print(f"Read data from text file: {data[:100]}")  # Show first 100 characters of data
        elif extension == '.csv':
            with open(filepath, newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                data = ' '.join([' '.join(row) for row in reader])
                print(f"Read data from CSV file: {data[:100]}")  # Show first 100 characters of data
        else:
            print("Unsupported file format.")
            return {}
    except Exception as e:
        print(f"Failed to read file {filepath}: {e}")
        return {}

    return analyze_text(data)


async def main():
    print("Starting the program...")
    test_mode = input("Enter test mode? (yes/no): ")

    if test_mode.lower() == 'yes':
        # Test data containing sensitive information
        data = {
            'https://testsite.com': 'Contact email: example@example.com, CC: 1234-5678-9123-4567, SSN: 123-45-6789.'
        }
        for url, content in data.items():
            findings = analyze_text(content)
            print(f"Findings for {url}: {findings}")

        # You can also simulate training the model here
    else:
        choice = input("Do you want to (1) scrape websites or (2) analyze a file? Enter 1 or 2: ")
        if choice == '1':
            urls = ["https://example.com", "https://example.org", "https://example.net"]
            data = await scrape_websites(urls)
            print("Scraped data:")
            for url, content in data.items():
                print(f"{url}: {content[:50]}...")

            for url, text in data.items():
                findings = analyze_text(text)
                print(f"Findings for {url}: {findings}")

            X, feature_names = create_dataset(data)
            if X is not None:
                print(f"Dataset: {X.toarray()}, Feature names: {feature_names}")
                y = [0, 1, 1]
                model = train_model(X, y)
                print(f"Trained model: {model}")
                prediction = model.predict(X[0])
                print(f"Prediction: {prediction}")
        elif choice == '2':
            filepath = input("Enter the path to the file you want to analyze: ")
            findings = analyze_file(filepath)
            if findings:
                print("Findings from file:")
                for category, matches in findings.items():
                    print(f"{category}: {matches}")
            else:
                print("No sensitive data found or empty file.")
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    asyncio.run(main())
