import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from together import Together
from pipeline import DocentPipeline
from PIL import Image
import io
import re 

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArtEvaluator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the ArtEvaluator with optional API key."""
        self.client = Together()
        if api_key:
            self.client.api_key = api_key
        
        self.required_fields = [
            'artist', 'title_of_work', 'date_created', 
            'location', 'style'
        ]

    
    
    def load_ground_truth(self, ground_truth_path: str) -> Dict[str, Any]:
        """
        Load ground truth data from JSON file.
        
        Args:
            ground_truth_path (str): Path to ground truth JSON file
            
        Returns:
            Dict[str, Any]: Ground truth data
        """
        try:
            with open(ground_truth_path, 'r') as f:
                data = json.load(f)

                # Convert 'date_created' to string
                data["date_created"] = str(data["date_created"])
                return data
        except Exception as e:
            logger.error(f"Error loading ground truth file: {str(e)}")
            raise

    def evaluate_response(self, vlm_response: Dict[str, Any], 
                         ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare VLM response with ground truth.
        
        Args:
            vlm_response (Dict[str, Any]): Response from VLM
            ground_truth (Dict[str, Any]): Ground truth data
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        results = {
            'matches': {},
            'mismatches': {},
            'missing_fields': []
        }
        
        # Check for missing fields
        for field in self.required_fields:
            if field not in vlm_response:
                results['missing_fields'].append(field)
        
        # Compare available fields
        for field in self.required_fields:
            if field in vlm_response and field in ground_truth:
                vlm_value = vlm_response[field].lower().strip()
                gt_value = ground_truth[field].lower().strip()
                
                if vlm_value == gt_value:
                    results['matches'][field] = vlm_value
                else:
                    results['mismatches'][field] = {
                        'vlm': vlm_value,
                        'ground_truth': gt_value
                    }
        
        # Calculate accuracy
        total_fields = len(self.required_fields)
        correct_fields = len(results['matches'])
        results['accuracy'] = correct_fields / total_fields
        
        return results
    def _extract_json(self,response):
      try:
          # Regex to find the first JSON object in the text
          json_match = re.search(r"\{.*\}", response, re.DOTALL)
          if json_match:
              # Load and return the JSON object
              return json.loads(json_match.group())
          else:
              raise ValueError("No valid JSON found.")
      except json.JSONDecodeError as e:
          raise ValueError(f"Error decoding JSON: {e}")
    def evaluate_response_llm_judge(self, vlm_response: Dict[str, Any], 
                              ground_truth: Dict[str, Any]) -> Dict[str, Any]:
      """
      Use Gemma-2B to judge if VLM responses are semantically similar to ground truth
      using a single API call for all fields.
      
      Args:
          vlm_response (Dict[str, Any]): Response from VLM
          ground_truth (Dict[str, Any]): Ground truth data
          
      Returns:
          Dict[str, Any]: Evaluation results with LLM judgments
      """
      results = {
          'correct_fields': {},
          'incorrect_fields': {},
          'missing_fields': [],
          'field_scores': {}
      }
      
      # First collect missing fields
      for field in self.required_fields:
          if field not in vlm_response or field not in ground_truth:
              results['missing_fields'].append(field)
              results['field_scores'][field] = 0
      
      # Prepare comparison data for available fields
      fields_to_compare = [
          field for field in self.required_fields 
          if field not in results['missing_fields']
      ]
      
      if not fields_to_compare:
          results['accuracy'] = 0
          return results
          
      # Create a structured prompt for all fields
      prompt = """Compare the following pairs of values and determine if they are semantically equivalent or similar enough to be considered correct.

      Consider:
      - Names may have slight spelling variations or abbreviations
      - Dates may be in different formats but represent the same time
      - Locations may use historical or modern names
      - Styles may use related terms or subcategories

      For each field, output only a 1 if the values are equivalent or similar enough to be considered correct, or 0 if they are clearly different or incorrect.

      Fill out the results in this exact JSON format, nothing else:
      Fill out the proper json format, with commas following each value for score,
      {
          "field_scores": {
              "artist": score,
              "title_of_work": score,
              "date_created": score,
              "location": score,
              "style": score,
          }
      }
      where score is either 0 or 1.  Do not forget closing brackets } 
      Here are the pairs to compare:
      """
      
      # Add each field comparison to the prompt
      for field in fields_to_compare:
          prompt += f"\n{field}:\n"
          prompt += f"VLM Response: {vlm_response[field]}\n"
          prompt += f"Ground Truth: {ground_truth[field]}\n"

      try:
          # Get judgment from Mixtral for all fields at once
          response = self.client.chat.completions.create(
              model="google/gemma-2b-it",
              messages=[
                  {"role": "user", "content": prompt}
              ],
              temperature=0
          )
          # Parse the JSON response
          print(self._extract_json(response.choices[0].message.content.replace("`","".strip())))
          scores = self._extract_json(response.choices[0].message.content.replace("`","".strip()))['field_scores']
          
          # Process results for each field
          for field in fields_to_compare:
              score = int(scores.get(field, 0))
              results['field_scores'][field] = score
              
              if score == 1:
                  results['correct_fields'][field] = {
                      'vlm': vlm_response[field],
                      'ground_truth': ground_truth[field]
                  }
              else:
                  results['incorrect_fields'][field] = {
                      'vlm': vlm_response[field],
                      'ground_truth': ground_truth[field]
                  }
                  
      except Exception as e:
          logger.error(f"Error getting LLM judgment: {str(e)}")
          # In case of error, mark all fields as incorrect
          for field in fields_to_compare:
              results['field_scores'][field] = 0
              results['incorrect_fields'][field] = {
                  'vlm': vlm_response[field],
                  'ground_truth': ground_truth[field]
              }
      
      # Calculate overall accuracy
      total_fields = len(self.required_fields)
      correct_fields = sum(results['field_scores'].values())
      results['accuracy'] = correct_fields / total_fields
      
      return results

def main():
    """Main function to run the evaluation."""
    try:
        # Initialize evaluator
        evaluator = ArtEvaluator()
        docent = DocentPipeline()
        # Define paths
        image_path = "input_image.png"
        ground_truth_path = "ground_truth.json"
        
        # Verify files exist
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not Path(ground_truth_path).exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
        
        # Get VLM response
        logger.info("Getting VLM response...")
        vlm_response = docent.get_vlm_response(image_path)
        
        # Load ground truth
        logger.info("Loading ground truth...")
        ground_truth = evaluator.load_ground_truth(ground_truth_path)
        
        # Evaluate response
        logger.info("Evaluating response...")
        results = evaluator.evaluate_response(vlm_response, ground_truth)
        
        print("Model Response: \n")
        print(vlm_response)
        print("Ground Truth: \n")
        print(ground_truth)
        # Print results
        print("\nEvaluation Results:")
        print(f"Accuracy: {results['accuracy']*100:.2f}%")
        
        print("\nCorrect Matches:")
        for field, value in results['matches'].items():
            print(f"- {field}: {value}")
            
        print("\nMismatches:")
        for field, values in results['mismatches'].items():
            print(f"- {field}:")
            print(f"  VLM: {values['vlm']}")
            print(f"  Ground Truth: {values['ground_truth']}")
            
        if results['missing_fields']:
            print("\nMissing Fields:")
            for field in results['missing_fields']:
                print(f"- {field}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()