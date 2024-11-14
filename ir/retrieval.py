from image_retrieval import ImageRetrieval
from wikipedia_retrieval import WikipediaRetrieval

class InformationRetrieval:
  def __init__(self, dataset_dir: str, json_dir: str):
    self.image_retrieval = ImageRetrieval(dataset_dir, json_dir)
    self.wikipedia_retrieval = WikipediaRetrieval()
  
  def get_context(self, image_path: str):
    _, metadata = self.image_retrieval.retrieve_most_similar_image(image_path)
    wiki_context = self.wikipedia_retrieval.search_from_json(metadata)
    return metadata, wiki_context
  
# Usage example
dataset_dir = "../scrape/data/images"
json_dir = "../scrape/data/json"
image_path = "../scrape/data/images/robert-delaunay_rhythm-1.jpg"
ir = InformationRetrieval(dataset_dir, json_dir)
metadata, wiki_context = ir.get_context(image_path)
print(metadata, wiki_context)