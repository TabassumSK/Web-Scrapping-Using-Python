# Twitter Scraper API :
# Scrapingdog provides a dedicated Twitter Scraper API which can be used for scraping Twitter at scale. Let’s sign up for the trial and see how it actually works.
# After signup, you will be redirected to your dashboard. You will find an X Scraper on the left. 
# Suppose, I want to scrape this tweet. I can directly use the dashboard to scrape it.
# In the above Screenshot, as you can see you got details like tweet text, number of likes, number of comments, etc.
# You can also copy the ready-to-use Python code on the right and paste it directly into your development environment.

import requests

api_key = "your-api-key"
url = "https://api.scrapingdog.com/twitter"

params = {
    "api_key": api_key,
    "url": "https://x.com/MedvedevRussiaE/status/1902119607478939707",
    "parsed": "true"
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Request failed with status code: {response.status_code}")
