import os
from typing import Dict, Any, Optional
import logging
from ir import InformationRetrieval
from dotenv import load_dotenv
from together import Together

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocentPipeline:
  def __init__(self, dataset_dir: str, json_dir: str, api_key: Optional[str] = None):
    """Initialize the ArtEvaluator with optional API key."""
    self.client = Together()
    if api_key:
      self.client.api_key = api_key
    
    self.ir = InformationRetrieval(dataset_dir, json_dir)
    
    self.required_fields = [
      'artist', 'title_of_work', 'date_created', 
      'location', 'style'
    ]
  
  def run(self, image_path):
    metadata, context = self.ir.get_context(image_path)
    return self.get_final_response(metadata, context)

  def get_final_response(self, metadata: str, context: str):
    prompt = f"""
    You are a docent at an art museum. We have just come across this painting on a tour. Write an accessible, engaging summary for the art historical context of this work, using the following factual information:

    Essential Facts:
    <essential_facts>{metadata}</essential_facts>

    Context:
    <context>{context}</context>

    Your response should only use the factual information provided. Please use all facts present in the provided Essential Facts JSON. Keep your response to 10 sentences.
    """

    messages = [
        {
          "role": "user",
          "content": [
          {
            "type": "text",
            "text": prompt
          }
        ]
        }
      ]
    
    response = self.client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        messages=messages
      )
    
    return response.choices[0].message.content

if __name__ == "__main__":
  dataset_dir = "scrape/data_v3/images"
  json_dir = "scrape/data_v3/json"
  pipeline = DocentPipeline(dataset_dir, json_dir, os.getenv('TOGETHER_API_KEY'))
  image_path = "scrape/data_v3/images/caravaggio_medusa-1597-1.jpg"
  res = pipeline.run(image_path)
  print(res)