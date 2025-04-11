import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import csv

# Function to scrape headlines and publication dates from European Parliament News
def scrape_europarl_headlines(last_months=6):
    url = "https://www.europarl.europa.eu/news/en"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # Get the current date and calculate the date for 6 months ago
    current_date = datetime.now()
    six_months_ago = current_date - timedelta(days=last_months * 30)  # Approximation of 6 months
    
    headlines = []
    page_num = 1  # Start from the first page
    total_articles_found = 0  # Accumulative counter for articles
    
    print("Starting the scraping process...\n")
    
    while True:
        # Send HTTP request to get page content
        print(f"Scraping page {page_num}...")
        response = requests.get(url, headers=headers, params={"page": page_num})
        
        if response.status_code != 200:
            print(f"Failed to retrieve page {page_num}. Status code: {response.status_code}")
            break
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find all articles on the page
        articles = soup.find_all("article")
        
        if not articles:
            print(f"No more articles found on page {page_num}.")
            break  # If no articles found, stop
        
        print(f"Found {len(articles)} articles on page {page_num}. Processing...\n")
        
        for article in articles:
            headline_tag = article.find("h3")
            date_tag = article.find("time")  # Look for the date of the article
            
            if headline_tag and date_tag:
                headline = headline_tag.get_text(strip=True)
                date_str = date_tag.get("datetime")  # Get the 'datetime' attribute for the date
                if headline and date_str:
                    article_date = datetime.fromisoformat(date_str)  # Convert the date to a datetime object
                    
                    # Check if the article is within the last 6 months
                    if article_date >= six_months_ago:
                        headlines.append((headline, article_date))
                        total_articles_found += 1  # Increment the counter for each valid article
                    else:
                        # If we find an article older than 6 months, stop the loop
                        print("\nReached articles older than 6 months. Stopping scraping.")
                        print(f"Total articles found: {total_articles_found}\n")
                        return headlines
        
        print(f"Total articles found so far: {total_articles_found}")
        page_num += 1  # Go to the next page

    return headlines

# Save results to CSV
def save_headlines_to_csv(headlines):
    with open("europarl_headlines_last_6_months.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Headline", "Date"])  # Writing headers
        for headline, date in headlines:
            writer.writerow([headline, date.strftime("%Y-%m-%d")])  # Writing each headline and its date

# Scrape the headlines from the last 6 months
headlines = scrape_europarl_headlines()

# Print the first few headlines for preview
print("\nScraping complete! Here are the first 10 headlines:\n")
for i, (headline, date) in enumerate(headlines[:10]):  # Print top 10 for preview
    print(f"{i+1}. {headline} (Date: {date.strftime('%Y-%m-%d')})")

# Save the results to CSV
save_headlines_to_csv(headlines)

# Confirmation message
print(f"\nAll headlines from the last 6 months have been saved to 'europarl_headlines_last_6_months.csv'.")
