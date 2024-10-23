import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from together import Together
from PIL import Image
import io

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
            "location": YOUR ANSWER,he
            "style": YOUR ANSWER
        }
        where style is the art style where the work came from, artist is the FULL NAME of the painter who created the work,  
        title_of_work is the FULL title of the work, date_created is the year when the work was created, in the format: YEAR and location is where the work was created, in the format: TOWN, COUNTRY. Only output the exact json format and nothing else."""
        
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
                return json.load(f)
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

def main():
    """Main function to run the evaluation."""
    try:
        # Initialize evaluator
        evaluator = ArtEvaluator()
        
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
        vlm_response = evaluator.get_vlm_response(image_path)
        
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