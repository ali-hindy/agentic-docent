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
  
  def encode_image(self, image_path: str) -> str:
    """
    Encode image to base64 string.
    
    Args:
      image_path (str): Path to the image file
        
    Returns:
      str: Base64 encoded image string
    """
    try:
      with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
      logger.error(f"Error encoding image: {str(e)}")
      raise
        
  def get_vlm_response(self, image_path: str) -> Dict[str, Any]:
    """
    Get VLM response for an image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Dict[str, Any]: Parsed JSON response from the VLM
    """
    prompt = """You are an art docent at a museum. Given the following image, fill out the information about the image in json format:
    {
        "artist": YOUR ANSWER,
        "title_of_work": YOUR ANSWER,
        "date_created": YOUR ANSWER,
        "location": YOUR ANSWER,
        "style": YOUR ANSWER
    }
    where style is the art style where the work came from, artist is the FULL NAME of the painter who created the work,  
    title_of_work is the FULL title of the work, date_created is the year when the work was created, in the format: YEAR and location is where the work was created, in the format: TOWN, COUNTRY. Only output the exact json format and nothing else. If the information is unknown, write unknown.
    
    Make sure to write proper json format:
    """
    
    try:
      # Encode image
      image_data = self.encode_image(image_path)
      
      # Create the message with image
      messages = [
        {
          "role": "user",
          "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/png;base64,{image_data}"
              }
          },
          {
            "type": "text",
            "text": prompt
          }
        ]
        }
      ]
        
      # Get response from VLM
      response = self.client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        messages=messages
      )
        
      # Parse the JSON response
      response_text = response.choices[0].message.content
      return json.loads(response_text)
        
    except json.JSONDecodeError as e:
      logger.error(f"Error parsing VLM response as JSON: {str(e)}")
      raise
    except Exception as e:
      logger.error(f"Error getting VLM response: {str(e)}")
      raise
  