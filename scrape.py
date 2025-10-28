import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
books_data = []

for page in range(1, 6):  # scrape first 5 pages
    url = BASE_URL.format(page)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    books = soup.find_all('article', class_='product_pod')
    for book in books:
        title = book.h3.a['title']
        price = book.find('p', class_='price_color').text
        availability = book.find('p', class_='instock availability').text.strip()
        rating = book.p['class'][1]  # rating like 'Three'
        books_data.append([title, price, rating, availability])
    
    print(f"✅ Scraped Page {page}")
    time.sleep(2)  # be polite to server

df = pd.DataFrame(books_data, columns=['Title', 'Price', 'Rating', 'Availability'])
df.to_csv('books_data.csv', index=False)
print("Data saved to books_data.csv successfully!")


# import csv
# with open('quotes.csv', 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['Quote', 'Author'])
#     for i in range(len(quotes)):
#         writer.writerow([quotes[i].text, authors[i].text])
