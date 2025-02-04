import requests
from bs4 import BeautifulSoup
import csv
from time import sleep
from urllib.parse import urljoin, urlparse
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def is_valid_url(url):
    """
    Check if URL is valid and belongs to pwani.net domain
    """
    if not url:
        return False
    
    # Skip social media, email, phone, and map links
    invalid_domains = ['facebook.com', 'instagram.com', 'linkedin.com', 
                      'tiktok.com', 'twitter.com', 'google.com', 
                      'mailto:', 'tel:', 'clifford.co.ke']
    
    return ('pwani.net' in url and 
            not any(domain in url for domain in invalid_domains))

def clean_text(text):
    """
    Clean extracted text by removing extra whitespace and empty lines
    """
    if not text:
        return ""
    
    # Remove extra whitespace and newlines
    cleaned = ' '.join(text.split())
    return cleaned

def scrape_page_content(url, visited_urls=None):
    """
    Scrape all text content from a single page
    """
    if visited_urls is None:
        visited_urls = set()
    
    if url in visited_urls:
        return None
    
    visited_urls.add(url)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Add delay to be respectful to the server
        sleep(2)
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()
        
        # Extract main content
        content = {
            'url': url,
            'title': clean_text(soup.title.string) if soup.title else '',
            'headings': [clean_text(h.text) for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])],
            'paragraphs': [clean_text(p.text) for p in soup.find_all('p') if clean_text(p.text)],
            'links': []
        }
        
        # Extract links for further scraping
        for link in soup.find_all('a', href=True):
            full_url = urljoin(url, link['href'])
            if is_valid_url(full_url):
                content['links'].append(full_url)
        
        return content
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def save_to_csv(data, filename):
    """
    Save scraped data to a CSV file.
    """
    if not data:
        return
    
    # Get all unique keys from the data
    headers = list(data.keys())
    
    # Find the maximum length of all lists in the data
    max_length = max(len(str(values)) if isinstance(values, str) else len(values) for values in data.values())
    
    # Write to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        # Write data rows
        for i in range(max_length):
            row = []
            for header in headers:
                if isinstance(data[header], str):
                    row.append(data[header])
                else:
                    row.append(data[header][i] if i < len(data[header]) else '')
            writer.writerow(row)

def main():
    # Start with the provided URLs
    urls = [
        "https://pwani.net/cooking-oils/",
        "https://pwani.net/personal-care/",
        "https://pwani.net/home-care/",
        "https://pwani.net/about-us/",
        "https://pwani.net/recipes/",
        "https://pwani.net/media-bulletin/",
        "https://pwani.net/contact-us/"
    ]
    
    visited_urls = set()
    all_content = []
    
    print("Starting content scraping...")
    
    # Scrape each URL
    for url in urls:
        if is_valid_url(url) and url not in visited_urls:
            print(f"Scraping: {url}")
            content = scrape_page_content(url, visited_urls)
            if content:
                all_content.append(content)
                
                # Add new found links to scrape
                for new_url in content['links']:
                    if new_url not in visited_urls:
                        urls.append(new_url)
    
    # Save results
    save_to_csv(all_content, 'pwani_content.csv')
    print(f"\nScraping completed. Processed {len(visited_urls)} pages.")
    print("Results saved to pwani_content.csv")

if __name__ == "__main__":
    main()