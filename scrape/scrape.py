import os
import requests
import json
from bs4 import BeautifulSoup
import weasyprint
from dotenv import load_dotenv

load_dotenv()

# Model:
# {
#   "artist": "Vincent Van Gogh",
#   "title_of_work": "The Starry Night",
#   "date_created":"1889", 
#   "location": "Saint-Rémy-de-Provence, France",
#   "style": "Post-Impressionism"
# }

# Helper function to scrape URIs and grab images for each painting
def scrape_paintings(image_dir):
    list_url = "https://www.wikiart.org/en/api/2/MostViewedPaintings"
    painting_data = {}
    has_more = True

    session_key = get_wikiart_session()

    count = 0
    while (has_more and count < 2):
        count += 1
        try:
            pagination_token = ""
            payload = {
                'paginationToken' : pagination_token,
                'authSessionKey' : session_key
            }
            response = requests.get(list_url, params=payload)
            if response.status_code == 200:
                jsonResponse = response.json()
                has_more = jsonResponse["hasMore"]
                pagination_token = jsonResponse["paginationToken"]
                
                for painting in jsonResponse["data"]:
                    painting_data[os.path.join(painting["artistUrl"], painting["url"])] = {
                        "artist" : painting["artistName"],
                        "title_of_work" : painting["title"],
                        "date_created" : painting["completitionYear"] # No, this is not a typo
                    }
                    save_painting_image(painting, image_dir)

        except Exception as e:
            print(f"Error fetching page URLs: {e}")
    return painting_data

# Generates API session token
def get_wikiart_session():
    payload = {
        'accessCode' : os.getenv('WIKIART_ACCESS_CODE'),
        'secretCode' : os.getenv('WIKIART_SECRET_CODE')
    }
    login_url = "https://www.wikiart.org/en/Api/2/login"
    response = requests.get(login_url, params=payload)
    return response.json()["SessionKey"]

# Save image given painting json response
def save_painting_image(painting, image_dir):
    img = requests.get(painting["image"])
    save_path = os.path.join(image_dir, painting["artistUrl"] + "_" + painting["url"] + ".jpg")

    if img.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(img.content)
        print("Saved image: " + save_path)

# Save HTML page as PDF
def save_page_as_pdf(uri, output_dir):
    base_url = "https://www.wikiart.org/en/"
    url = base_url + uri
    try:
        output_path = os.path.join(output_dir, uri.replace("/", "_") + ".pdf")
        doc = weasyprint.HTML(url=url).render()
        doc.copy(doc.pages[3:4]).write_pdf(output_path) # For these renders, the third page contains our target data
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error saving PDF for {url}: {e}")

# Scrape additional data directly from page and write to JSON
def write_data_from_page(uri, painting_data, json_dir):
    base_url = "https://www.wikiart.org/en/"
    response = requests.get(base_url + uri)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        for li in soup.find_all('li'):
            if li.find('s') and "Style" in li.find('s').text:
                painting_data[uri]["style"] = li.find('a').text.strip()
            if li.find('s') and "Location" in li.find('s').text:
                painting_data[uri]["location"] = li.find('span').text.strip()

    save_path = os.path.join(json_dir, uri.replace("/", "_") + ".json")
    with open(save_path, 'w') as f:
        json.dump(painting_data[uri], f, indent=4)
    print("Saved json: " + save_path)

# Main scraper function
def scrape_wikiart(output_dir):
    # Get all page URLs from the website
    image_dir = os.path.join(output_dir, "images")
    painting_data = scrape_paintings(image_dir)

    pdf_dir = os.path.join(output_dir, "pdfs")
    json_dir = os.path.join(output_dir, "json")
    
    # Iterate over each URL and save as PDF
    for uri in painting_data.keys():
        write_data_from_page(uri, painting_data, json_dir)
        save_page_as_pdf(uri, pdf_dir)

if __name__ == "__main__":
    output_dir = "./data"
    scrape_wikiart(output_dir)