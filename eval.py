import base64
import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
from together import Together

from pipeline import DocentPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ArtEvaluator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the ArtEvaluator with optional API key."""
        self.client = Together()
        if api_key:
            self.client.api_key = api_key

        self.required_fields = ["artist", "title_of_work", "date_created", "style"]

    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string.

        Args:
        image_path (str): Path to the image file

        Returns:
        str: Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
        raise

    def get_vlm_response_baseline(self, image_path: str) -> Dict[str, Any]:
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
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Get response from VLM
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                messages=messages,
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

    def get_baseline_full_response(self, image_path: str) -> str:
        """
        Get a complete art analysis response from the Llama baseline model in a single prompt,
        focusing on artist, style, and title while including visual analysis.

        Args:
            image_path (str): Path to the image file

        Returns:
            str: A natural language response analyzing the artwork
        """
        prompt = """You are an art historian being asked about a painting. Write an accessible, engaging 
        summary about this artwork in a single paragraph (under 100 words). Your response should include:
        1. The artist's name
        2. The title of the work
        3. The art style or movement
        4. A brief description of the most striking visual elements you can see

        Write naturally, as if speaking to an interested museum visitor. If you're unsure about any details,
        acknowledge this uncertainty but provide your best assessment based on the visual evidence."""

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
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Get response from VLM
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                messages=messages,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating baseline response: {str(e)}")
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
            with open(ground_truth_path, "r") as f:
                data = json.load(f)

                # Convert 'date_created' to string
                data["date_created"] = str(data["date_created"])
                return data
        except Exception as e:
            logger.error(f"Error loading ground truth file: {str(e)}")
            raise

    def evaluate_response(
        self, vlm_response: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare VLM response with ground truth.

        Args:
            vlm_response (Dict[str, Any]): Response from VLM
            ground_truth (Dict[str, Any]): Ground truth data

        Returns:
            Dict[str, Any]: Evaluation results
        """
        results = {"matches": {}, "mismatches": {}, "missing_fields": []}

        # Check for missing fields
        for field in self.required_fields:
            if field not in vlm_response:
                results["missing_fields"].append(field)

        # Compare available fields
        for field in self.required_fields:
            if field in vlm_response and field in ground_truth:
                vlm_value = vlm_response[field].lower().strip()
                gt_value = ground_truth[field].lower().strip()

                if vlm_value == gt_value:
                    results["matches"][field] = vlm_value
                else:
                    results["mismatches"][field] = {
                        "vlm": vlm_value,
                        "ground_truth": gt_value,
                    }

        # Calculate accuracy
        total_fields = len(self.required_fields)
        correct_fields = len(results["matches"])
        results["accuracy"] = correct_fields / total_fields

        return results

    def _extract_json(self, response):
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

    def evaluate_response_llm_judge(
        self, vlm_response: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            "correct_fields": {},
            "incorrect_fields": {},
            "missing_fields": [],
            "field_scores": {},
        }

        # First collect missing fields
        for field in self.required_fields:
            if field not in vlm_response or field not in ground_truth:
                results["missing_fields"].append(field)
                results["field_scores"][field] = 0

        # Prepare comparison data for available fields
        fields_to_compare = [
            field
            for field in self.required_fields
            if field not in results["missing_fields"]
        ]

        if not fields_to_compare:
            results["accuracy"] = 0
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
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            # Parse the JSON response
            print(
                self._extract_json(
                    response.choices[0].message.content.replace("`", "".strip())
                )
            )
            scores = self._extract_json(
                response.choices[0].message.content.replace("`", "".strip())
            )["field_scores"]

            # Process results for each field
            for field in fields_to_compare:
                score = int(scores.get(field, 0))
                results["field_scores"][field] = score

                if score == 1:
                    results["correct_fields"][field] = {
                        "vlm": vlm_response[field],
                        "ground_truth": ground_truth[field],
                    }
                else:
                    results["incorrect_fields"][field] = {
                        "vlm": vlm_response[field],
                        "ground_truth": ground_truth[field],
                    }

        except Exception as e:
            logger.error(f"Error getting LLM judgment: {str(e)}")
            # In case of error, mark all fields as incorrect
            for field in fields_to_compare:
                results["field_scores"][field] = 0
                results["incorrect_fields"][field] = {
                    "vlm": vlm_response[field],
                    "ground_truth": ground_truth[field],
                }

        # Calculate overall accuracy
        total_fields = len(self.required_fields)
        correct_fields = sum(results["field_scores"].values())
        results["accuracy"] = correct_fields / total_fields

        return results

    def evaluate_text_response_llm_judge(
        self, text_response: str, ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Gemma-2B to judge if a text paragraph contains information matching the ground truth fields
        using a single API call for all fields.

        Args:
            text_response (str): Text paragraph response from model
            ground_truth (Dict[str, Any]): Ground truth data with required fields

        Returns:
            Dict[str, Any]: Evaluation results with LLM judgments
        """
        results = {
            "correct_fields": {},
            "incorrect_fields": {},
            "missing_fields": [],
            "field_scores": {},
            "extracted_values": {},  # New field to store extracted values from text
        }

        # Check for missing ground truth fields
        for field in self.required_fields:
            if field not in ground_truth:
                results["missing_fields"].append(field)
                results["field_scores"][field] = 0

        fields_to_check = [
            field
            for field in self.required_fields
            if field not in results["missing_fields"]
        ]

        if not fields_to_check:
            results["accuracy"] = 0
            return results

        # Create a structured prompt for analyzing the text
        prompt = f"""Analyze the following text paragraph and determine if it contains information matching each required field.
        For each field, determine:
        1. If the information is present in the text
        2. Extract the specific value if present
        3. Compare it with the ground truth value
        
        Consider:
        - Names may have slight spelling variations or abbreviations
        - Dates may be in different formats but represent the same time
        - Locations may use historical or modern names
        - Styles may use related terms or subcategories

        Text to analyze:
        {text_response}

        For each field, output in this exact JSON format, nothing else:
        """
        prompt += """
        {
            "field_results": {{
                "field_name": {{
                    "present": 0 or 1,
                    "extracted_value": "extracted text or null if not found",
                    "matches_ground_truth": 0 or 1,
                }},
                ...
            }}
        }}

        Fields to check and their ground truth values:
        """

        # Add each field and its ground truth value to the prompt
        for field in fields_to_check:
            prompt += f"\n{field}: {ground_truth[field]}"

        try:
            # Get judgment from Gemma for text analysis
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            # Parse the JSON response
            analysis = self._extract_json(
                response.choices[0].message.content.replace("`", "".strip())
            )["field_results"]
            # Process results for each field
            for field in fields_to_check:
                field_result = analysis.get(field, {})
                is_present = int(field_result.get("present", 0))
                matches_truth = int(field_result.get("matches_ground_truth", 0))
                extracted_value = field_result.get("extracted_value")

                results["field_scores"][field] = matches_truth
                results["extracted_values"][field] = extracted_value

                if matches_truth == 1:
                    results["correct_fields"][field] = {
                        "extracted": extracted_value,
                        "ground_truth": ground_truth[field],
                    }
                else:
                    results["incorrect_fields"][field] = {
                        "extracted": extracted_value,
                        "ground_truth": ground_truth[field],
                        "present_in_text": bool(is_present),
                    }

        except Exception as e:
            logger.error(f"Error getting LLM judgment: {str(e)}")
            # In case of error, mark all fields as incorrect
            for field in fields_to_check:
                results["field_scores"][field] = 0
                results["incorrect_fields"][field] = {
                    "extracted": None,
                    "ground_truth": ground_truth[field],
                    "present_in_text": False,
                }

        # Calculate overall accuracy
        total_fields = len(self.required_fields)
        correct_fields = sum(results["field_scores"].values())
        results["accuracy"] = correct_fields / total_fields

        return results

    def evaluate_style_response_llm_judge(self, text_response: str) -> Dict[str, Any]:
        """
        Use an LLM to evaluate stylistic aspects of a text response including helpfulness,
        informativeness, and engagement level.

        Args:
            text_response (str): Text response to evaluate

        Returns:
            Dict[str, Any]: Evaluation results with LLM judgments for style criteria
        """
        results = {"style_scores": {}, "style_feedback": {}, "overall_style_score": 0}

        # Define style criteria to evaluate
        style_criteria = {
            "helpfulness": "evaluates if the response has a helpful tone, showing willingness to assist and provide solutions",
            "informative": "evaluates if the response effectively communicates information in a clear and educational manner",
            "engaging": "evaluates if the response maintains interest through active and dynamic communication",
        }

        prompt = f"""Analyze the following text response and evaluate its stylistic qualities.
        For each criterion, determine if the response meets the standard (1) or not (0).
        Provide specific evidence from the text to support your judgment.
        
        Text to analyze:
        {text_response}
        """
        prompt += """
        For each criterion, output in this exact JSON format, nothing else:
        {{
            "style_results": {{
                "criterion_name": {{
                    "score": 0 or 1,
                    "evidence": "specific examples from text supporting the judgment"
                }},
                ...
            }}
        }}

        Style criteria to evaluate:
        """

        # Add each criterion and its description to the prompt
        for criterion, description in style_criteria.items():
            prompt += f"\n{criterion}: {description}"

        try:
            # Get judgment from LLM for style analysis
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            # Parse the JSON response
            analysis = self._extract_json(
                response.choices[0].message.content.replace("`", "").strip()
            )["style_results"]

            # Process results for each criterion
            for criterion in style_criteria:
                criterion_result = analysis.get(criterion, {})
                score = int(criterion_result.get("score", 0))
                evidence = criterion_result.get("evidence", "")

                results["style_scores"][criterion] = score
                results["style_feedback"][criterion] = evidence

            # Calculate overall style score (average of all criteria)
            total_score = sum(results["style_scores"].values())
            results["overall_style_score"] = total_score / len(style_criteria)

        except Exception as e:
            logger.error(f"Error getting LLM style judgment: {str(e)}")
            # In case of error, mark all criteria as 0
            for criterion in style_criteria:
                results["style_scores"][criterion] = 0
                results["style_feedback"][criterion] = "Error during evaluation"
            results["overall_style_score"] = 0

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
        vlm_response = docent.get_vlm_response_baseline(image_path)

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
        for field, value in results["matches"].items():
            print(f"- {field}: {value}")

        print("\nMismatches:")
        for field, values in results["mismatches"].items():
            print(f"- {field}:")
            print(f"  VLM: {values['vlm']}")
            print(f"  Ground Truth: {values['ground_truth']}")

        if results["missing_fields"]:
            print("\nMissing Fields:")
            for field in results["missing_fields"]:
                print(f"- {field}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
