import requests
from bs4 import BeautifulSoup

def get_paper_titles(query, api_key, search_engine_id):
    url = "https://www.googleapis.com/customsearch/v1"
    
    params = {
        'key': api_key,
        'cx': search_engine_id,
        'q': query,
        'num': 10,
        'siteSearch': 'arxiv.org',
    }
    
    response = requests.get(url, params=params)
    results = response.json()
    
    links = [item['link'] for item in results.get('items', []) if 'arxiv.org' in item['link']]
    
    titles = []
    for link in links:
        paper_page = requests.get(link)
        soup = BeautifulSoup(paper_page.content, 'html.parser')
        title_tag = soup.find('title')
        if title_tag:
            titles.append(title_tag.text.split(']')[1].strip())  # Extracting title from the format [arXiv:id] Title
    
    return titles

# # Replace with your API Key and Search Engine ID
# API_KEY = "AIzaSyC230uV-vbwvHjtIxqGdXdX-0kFCrlp78o"
# SEARCH_ENGINE_ID = "930c007e311a24f1c"

# query = input("Enter your query: ")
# titles = get_paper_titles(query, API_KEY, SEARCH_ENGINE_ID)
# for title in titles:
#     print(title)
