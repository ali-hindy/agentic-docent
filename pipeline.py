import os
from typing import Dict, Any, Optional, Literal
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
  def __init__(
      self, 
      dataset_dir: str, 
      json_dir: str, 
      api_key: Optional[str] = None,
      embedding_type: Literal["ResNet", "ColPali"] = "ResNet"
  ):
    self.client = Together()
    if api_key:
      self.client.api_key = api_key
    
    self.ir = InformationRetrieval(dataset_dir, json_dir, self.client, embedding_type=embedding_type)
  
  
  def run(self, image_path):
    metadata, context, exact_match = self.ir.get_context(image_path)
    return self.get_final_response(metadata, context, exact_match)

  def get_final_response(self, metadata: str, context: str, exact_match: bool):
    prompt = f"""
    You are an art historian being asked about a painting. Write an accessible, engaging summary for the art historical context of this work, using the following factual information:
    {"Begin your response by clarifying you couldn't identify the painting exactly, but that it appears similar to the following artist and style." if not exact_match else ""}
    Essential Facts:
    <essential_facts>{metadata}</essential_facts>

    Context:
    <context>{context}</context>

    Your response should only use the factual information provided. Please use all facts present in the provided Essential Facts JSON. Keep your response 5-8 sentences.
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
  dataset_dir = "./data_v3/images"
  json_dir = "./data_v3/json"
  pipeline = DocentPipeline(dataset_dir, json_dir, os.getenv('TOGETHER_API_KEY'))
  image_paths = [
    #"./data_v3/images/caravaggio_medusa-1597-1.jpg",
    "./ir/fenetre-ouverte.jpg"
  ]
  for path in image_paths:
    res = pipeline.run(path)
    print(res)