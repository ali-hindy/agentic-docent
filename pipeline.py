import os
from typing import Dict, Any, Optional, Literal
import logging
import base64
from ir import InformationRetrieval
from dotenv import load_dotenv
from together import Together
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

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
      embedding_type: Literal["ResNet", "CLIP"] = "ResNet",
      sim_threshold: float = 0.9
  ):
    self.client = Together()
    if api_key:
      self.client.api_key = api_key
    
    self.ir = InformationRetrieval(dataset_dir, json_dir, self.client, embedding_type=embedding_type, sim_threshold=sim_threshold)
  
  
  def run(self, image_path):
    metadata, context, exact_match = self.ir.get_context(image_path)
    return self.get_final_response(image_path, metadata, context, exact_match)

  def get_final_response(self, image_path: str, metadata: str, context: str, exact_match: bool):
    print("\nGenerating final response...")

    with open(image_path, "rb") as f:
      base64_image = base64.b64encode(f.read()).decode('utf-8')

    prompt = f"""
    You are an art historian being asked about a painting. Write an accessible, engaging summary for the art historical context and visual analysis of this work, using the following factual information:
    {"Before anything else, begin your response by clarifying you couldn't identify the painting, but that it appears somewhat similar to the following artist and style." if not exact_match else ""}
    Essential Facts:
    <essential_facts>{metadata}</essential_facts>

    Context:
    <context>{context}</context>

    Incorporate at least one specific phrase about the key visually significant aspects of the work.
    For art historical context, your response should only use the factual information provided. Visual analysis does not need to be grounded in the factual information.
    Please use all facts present in the provided Essential Facts JSON. Limit your entire response to one succint paragraph, under 100 words.
    """

    messages = [
        {
          "role": "user",
          "content": [
          {
            "type": "text",
            "text": prompt
          },
          {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            },
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
  pipeline = DocentPipeline(
    dataset_dir, 
    json_dir, 
    os.getenv('TOGETHER_API_KEY'),
    sim_threshold=0.9,
    embedding_type="CLIP"
  )
  image_paths = [
    # "cropped.jpg"
    # "mucha.jpg"
    "fenetre-ouverte.jpg"
  ]
  for path in image_paths:
    print(f"\nRunning docent pipeline for image {path}...")
    res = pipeline.run(path)
    print(f"FINAL RESPONSE FOR IMAGE {path}:\n\n{res}\n")