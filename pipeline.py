import json 
from typing import Dict, Any, Optional
import logging
import base64

from together import Together
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocentPipeline:
  def __init__(self, api_key: Optional[str] = None):
    """Initialize the ArtEvaluator with optional API key."""
    self.client = Together()
    if api_key:
      self.client.api_key = api_key
    
    self.required_fields = [
      'artist', 'title_of_work', 'date_created', 
      'location', 'style'
    ]
  

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
  