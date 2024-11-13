import requests
import json

class WikipediaRetrieval:
  def __init__(self):
    self.base_url = "https://en.wikipedia.org/w/api.php"

  # Get extracted wiki content for additional context from ground json data
  def search_from_json(self, json_data: dict):
    wiki_content = []
    
    keys = ["title_of_work", "artist", "style"]
    for k in keys:
       pages = self.get_pages(json_data[k])
       if pages is not None:
          wiki_content.append(self.get_extracts(pages))
    return wiki_content
  
  # Get Wikipedia pages given search query
  def get_pages(self, query: str):
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srwhat": "nearmatch", # Prevents false positives
        "srlimit": 1  # Limit to the top result for the best match
    }

    response = requests.get(self.base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data["query"]["search"]:
            best_match_title = data["query"]["search"][0]["title"]
            return best_match_title.replace(" ", "_")
        else:
           print(f"No match found for '{query}'")
           return None
    else:
        raise Exception("Error: {response.status_code}")
  
  # Get extracts from Wikipedia page given title
  def get_extracts(self, title: str):
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exsentences": 10, # Extract ten sentences
        "explaintext": True, # Plaintext instead of HTML
        "format": "json"
    }
    response = requests.get(self.base_url, params=params)
    data = response.json()
    
    pages = data["query"]["pages"]
    page_content = next(iter(pages.values()))["extract"]

    if "may refer to" in page_content:
        return None  # Indicate it's a disambiguation page
    return page_content

# Usage example
# wiki_retriever = WikipediaRetrieval()
# json_path = "../scrape/data/json/edvard-munch_the-scream-1893.json"
# print(wiki_retriever.search_from_json(json_path))