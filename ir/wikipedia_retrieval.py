import requests
import json

class WikipediaRetrieval:
  def __init__(self):
    self.base_url = "https://en.wikipedia.org/w/api.php"

  # Get extracted wiki content for additional context from ground json data
  def search_from_json(self, json_data: dict):
    print("\nSearching Wikipedia for additional context...")
    wiki_content = []
    
    for k in ["artist", "style", "title_of_work"]:
      if k in json_data:
        page = self.get_page_from_key(json_data, k)
        if page is not None:
            wiki_content.append(self.get_extracts(page))
            print(f"Extracted content from Wikipedia page: {page}")

    return wiki_content
  
  # Get Wikipedia pages from value queries given JSON + keys
  def get_page_from_key(self, json_data, key):
    query = json_data[key]
    if key == "title_of_work":
      # Attempt 1: Artist in page matching title
      title = self.get_page(query)
      extracts = None
      if title is not None:
        extracts = self.get_extracts(title)
      if extracts is not None:
        if json_data["artist"] in extracts:
          return title
      print(f"No Wikipedia page found for {query}")
      # Attempt 2: Page matching title + (<artist>)
      try:
        return self.get_page(f"{query} ({json_data['artist']})")
      except KeyError:
        print(f"No Wikipedia page found for {query} {json_data['artist']}).")
    else:
       return self.get_page(query)
    
  # Get Wikipedia pages given search query
  def get_page(self, query: str):
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srwhat": "text",
        "srlimit": 1  # Limit to the top result for the best match
    }

    response = requests.get(self.base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data["query"]["search"]:
            best_match = data["query"]["search"][0]
            title = best_match["title"]
            return title.replace(" ", "_")
        else:
           print(f"No Wikipedia page found for '{query}'")
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

if __name__ == "__main__":
  wiki_retriever = WikipediaRetrieval()
  data = {
    "artist": "Georgia O'Keeffe",
    "title_of_work": "City Night",
    "date_created": 1926,
    "style": "Precisionism"
  } 
  print(wiki_retriever.search_from_json(data))