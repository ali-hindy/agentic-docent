import os
import requests
import json
from bs4 import BeautifulSoup
import weasyprint
from dotenv import load_dotenv
import numpy as np
import pickle

load_dotenv()

def get_all_artists():
    list_url = "https://wikiart.org/en/api/2/UpdatedArtists"
    has_more = True
    session_key = get_wikiart_session()
    artist_ids = []
    pagination_token = ""
    count = 0
    while (has_more):
        try:
            full_url = f"{list_url}?paginationToken={pagination_token}&authSessionKey={session_key}"
            response = requests.get(full_url)
            if response.status_code == 200:
                count += 1
                print(f"Fetched page #{count} of artists")
                jsonResponse = response.json()
                has_more = jsonResponse["hasMore"]
                pagination_token = jsonResponse["paginationToken"]
                print(f"PToken: {pagination_token}")
                
                for artist in jsonResponse["data"]:
                    artist_ids.append(artist["id"])
            else:
                print(f"Response with status code {response.status_code}")
                print(response.json())

        except Exception as e:
            print(f"Error fetching page URLs: {e}")
    return artist_ids

# Helper function to fetch data for each of the most viewed paintings
# Note: API only returns the first 13892 results for some reason, after 234 pages
def get_most_viewed_paintings(painting_data={}, checkpoint_token=""):
    list_url = "https://www.wikiart.org/en/api/2/MostViewedPaintings"
    has_more = True

    session_key = get_wikiart_session()
    pagination_token = checkpoint_token
    count = 0
    while (has_more):
        try:
            full_url = f"{list_url}?paginationToken={pagination_token}&authSessionKey={session_key}"
            response = requests.get(full_url)
            if response.status_code == 200:
                count += 1
                print(f"Fetched page #{count} of most viewed paintings")
                jsonResponse = response.json()
                has_more = jsonResponse["hasMore"]
                pagination_token = jsonResponse["paginationToken"]
                print(f"PToken: {pagination_token}")
                
                for painting in jsonResponse["data"]:
                    painting_data[os.path.join(painting["artistUrl"], painting["url"])] = {
                        "artist" : painting["artistName"],
                        "title_of_work" : painting["title"],
                        "date_created" : painting["completitionYear"], # No, this is not a typo
                        "image" : painting["image"]
                    }
            else:
                print(f"Response with status code {response.status_code}")
                print(response.json())

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

    if response.status_code == 200:
        print("Successfully authenticated to WikiArt")
        return response.json()["SessionKey"]
    else:
        raise Exception("Error authenticating: status " + response.status_code)

# Save image given painting json response
def save_painting_image(uri, painting_data, output_dir):
    img = requests.get(painting_data[uri]["image"])
    save_path = os.path.join(output_dir, uri.replace("/", "_") + ".jpg")

    if img.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(img.content)
        print("Saved image: " + save_path)

# Save HTML page as PDF
def save_page_as_pdf(uri, output_dir):
    base_url = "https://www.wikiart.org/en/"
    url = base_url + uri
    while True:
        try:
            output_path = os.path.join(output_dir, uri.replace("/", "_") + ".pdf")
            # Define CSS for page size and content overflow handling
            css = weasyprint.CSS(string="""
                @page {
                    size: A1; /* Set page size to A4 or any other standard size */
                    margin: 1in; /* Adjust margins as needed */
                }
                body {
                    overflow: hidden; /* Prevent content from overflowing */
                }
                .content-section {
                    page-break-before: always; /* Force page breaks for specific sections if needed */
                }
            """)
            doc = weasyprint.HTML(url=url).render(stylesheets=[css])
            doc.copy(doc.pages[0:1]).write_pdf(output_path) # For these renders, the third page contains our target data
            print(f"Saved: {output_path}")
            break
        except Exception as e:
            print(f"Error saving PDF for {url}: {e}")
            print("Retrying...")

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
    del painting_data[uri]["image"]
    save_path = os.path.join(json_dir, uri.replace("/", "_") + ".json")
    with open(save_path, 'w') as f:
        json.dump(painting_data[uri], f, indent=4)
    print("Saved json: " + save_path)

# Sample paintings randomly with Zipfian distribution
def sample_paintings(painting_data, n_samples):
    n = len(painting_data.keys())
    print(f"Sampling {n_samples} out of {n} paintings.")
            
    # Generate a Zipfian distribution for ranks
    ranks = np.arange(1, n + 1)  # ranks from 1 to n
    zipf_probs = 1 / ranks  # inverse of rank, typical Zipf-like distribution
    zipf_probs /= zipf_probs.sum()  # normalize to get probabilities

    # Sample ranks based on Zipfian distribution
    print(n, n_samples)
    sampled_items = np.random.choice(list(painting_data.keys()), size=n_samples, replace=False, p=zipf_probs)

    return sampled_items

# Helpers for updating existing pickle
def update_most_viewed(data_path):
    checkpoint_token = os.getenv("CHECKPOINT_TOKEN")
    painting_data = {}
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            painting_data = pickle.load(f)
    painting_data = get_most_viewed_paintings(painting_data, checkpoint_token)
    with open(data_path, 'wb') as f:
        pickle.dump(painting_data, f)

def update_artists(data_path):
    artist_data = get_all_artists()
    with open(data_path, 'wb') as f:
        pickle.dump(artist_data, f)

# Sample from existing pickle
def sample_and_write(output_dir, data_path, n_samples):
    with open(data_path, 'rb') as f:
            painting_data = pickle.load(f)
    pdf_dir = os.path.join(output_dir, "pdfs")
    image_dir = os.path.join(output_dir, "images")
    json_dir = os.path.join(output_dir, "json")

    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    paintings = sample_paintings(painting_data, n_samples)

    # Scrape from each page and save PDF
    for i, uri in enumerate(paintings):
        print(f"Processing painting {i+1}/{len(paintings)}")
        save_painting_image(uri, painting_data, image_dir)
        write_data_from_page(uri, painting_data, json_dir)
        save_page_as_pdf(uri, pdf_dir)

if __name__ == "__main__":
    output_dir = "./data"
    data_path = 'artists.pkl'
    update_artists(data_path)
    #scrape_wikiart(data_path)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # n_samples = 500
    # sample_and_write(output_dir, data_path, n_samples)