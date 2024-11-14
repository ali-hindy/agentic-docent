from pathlib import Path
from eval import ArtEvaluator
from pipeline import DocentPipeline
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

def run_batch_evaluation(image_dir: str, json_dir: str, max_samples: int = 5) -> None:
    """
    Performs batch evaluation of artwork metadata extraction by comparing Vision Language Model (VLM) 
    outputs against ground truth labels, providing detailed accuracy metrics and error analysis.
    
    This function:
    1. Processes images and their corresponding ground truth JSON files
    2. Generates VLM predictions for each image using ArtEvaluator
    3. Compares predictions against ground truth using LLM-based similarity judgment
    4. Produces comprehensive evaluation metrics and error analysis
    
    Args:
        image_dir (str): Path to directory containing artwork images
            - Supported formats: .jpg
            - Images should be readable by PIL
            - Directory structure should be flat (no subdirectories)
        
        json_dir (str): Path to directory containing ground truth JSON files
            - Files must match image names (e.g., "image1.jpg" → "image1.json")
            - JSON format must contain fields: artist, title_of_work, date_created, location, style
            - Example JSON structure:
              {
                "artist": "Vincent van Gogh",
                "title_of_work": "The Starry Night",
                "date_created": "1889",
                "location": "Saint-Rémy-de-Provence, France",
                "style": "Post-Impressionism"
              }
        
        max_samples (int, optional): Maximum number of images to evaluate
            - Default: 5
            - Set to None to process all images in directory
    
    Returns:
        None. Results are printed to console and saved to files:
            - error_analysis.csv: Detailed breakdown of all prediction errors
    
    Outputs:
        1. Console Output:
           - Total number of images evaluated
           - Average accuracy across all images
           - Per-field accuracy breakdown
           - Detailed error analysis showing:
             * Field name
             * Image filename
             * VLM prediction
             * Ground truth value
           - Error distribution statistics
        
        2. CSV File (error_analysis.csv):
           - Columns: Field, Image, VLM Response, Ground Truth
           - One row per error instance
    
    Error Handling:
        - Skips images without corresponding JSON files
        - Continues processing if individual image evaluation fails
        - Logs errors to console
    
    Dependencies:
        - ArtEvaluator class with configured API access
        - pandas for error analysis
        - tqdm for progress tracking
        - pathlib for file handling
    
    Example Usage:
        >>> run_batch_evaluation(
        ...     image_dir="../data-small/images",
        ...     json_dir="../data-small/json",
        ...     max_samples=5
        ... )
        === EVALUATION RESULTS ===
        Total Images Evaluated: 5
        Average Accuracy: 85.33%
        
        === PER-FIELD ACCURACY ===
        artist: 90.00%
        title_of_work: 80.00%
        ...
        
        === ERROR ANALYSIS ===
        artist Errors (1 instances):
        Image: artwork1.jpg
        VLM Response: Pablo Picaso
        Ground Truth: Pablo Picasso
        ...
    
    Notes:
        - Requires valid API keys for VLM and LLM services
        - Processing time depends on API response times
        - Error analysis CSV is overwritten on each run
        - Memory usage scales with number of images processed
    """
    evaluator = ArtEvaluator()
    image_dir = Path(image_dir)
    json_dir = Path(json_dir)
    
    # Store results and error analysis
    results = {}
    error_analysis = defaultdict(list)
    
    # Track progress
    cnt = 0
    for image_path in tqdm(image_dir.glob("*.jpg")):
        if cnt >= max_samples:
            break
            
        gt_path = json_dir / f"{image_path.stem}.json"
        if gt_path.exists():
            try:
                vlm_response = evaluator.get_vlm_response_baseline(str(image_path))
                ground_truth = evaluator.load_ground_truth(str(gt_path))
                result = evaluator.evaluate_response_llm_judge(
                    vlm_response, ground_truth
                )
                results[image_path.name] = result
                
                # Collect errors for analysis
                for field, score in result['field_scores'].items():
                    if score == 0:
                        error_analysis[field].append({
                            'image': image_path.name,
                            'vlm': vlm_response.get(field, 'MISSING'),
                            'ground_truth': ground_truth.get(field, 'MISSING')
                        })
                
                cnt += 1
                
            except Exception as e:
                print(f"Error processing {image_path.name}: {str(e)}")
                continue
    
    # Calculate overall statistics
    total_acc = sum(val["accuracy"] for val in results.values())
    avg_accuracy = total_acc / len(results) if results else 0
    
    # Calculate per-field accuracy
    field_accuracies = defaultdict(list)
    for result in results.values():
        for field, score in result['field_scores'].items():
            field_accuracies[field].append(score)
    
    # Print results
   
    print("\n=== ERROR ANALYSIS ===")
    for field, errors in error_analysis.items():
        if errors:
            print(f"\n{field} Errors ({len(errors)} instances):")
            for error in errors:
                print(f"Image: {error['image']}")
                print(f"VLM Response: {error['vlm']}")
                print(f"Ground Truth: {error['ground_truth']}")
                print("-" * 50)
    
    # Create error summary DataFrame
    error_summary = []
    for field, errors in error_analysis.items():
        for error in errors:
            error_summary.append({
                'Field': field,
                'Image': error['image'],
                'VLM Response': error['vlm'],
                'Ground Truth': error['ground_truth']
            })
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"Total Images Evaluated: {len(results)}")
    print(f"Average Accuracy: {avg_accuracy:.2%}")
    
    print("\n=== PER-FIELD ACCURACY ===")
    for field, scores in field_accuracies.items():
        field_acc = sum(scores) / len(scores)
        print(f"{field}: {field_acc:.2%}")
    
    if error_summary:
        df = pd.DataFrame(error_summary)
        print("\n=== ERROR SUMMARY STATISTICS ===")
        print("\nErrors per Field:")
        print(df['Field'].value_counts())
        
        # Save error analysis to CSV
        output_path = "error_analysis_500.csv"
        df.to_csv(output_path, index=False)
        print(f"\nDetailed error analysis saved to: {output_path}")

if __name__ == "__main__":
    run_batch_evaluation(
        image_dir="./data_v2/images_handheld",
        json_dir="./data_v2/json",
        max_samples=500
    )