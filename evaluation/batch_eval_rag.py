from retriever import ImageRetrieval

from eval import ArtEvaluator
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

def run_batch_evaluation(image_dir: str, json_dir: str, max_samples: int = 5) -> None:
    image_retriever = ImageRetrieval(image_dir, json_dir)
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
        
        try:
            similar_image_path, metadata = image_retriever.retrieve_most_similar_image(str(image_path))
            
            vlm_response = metadata#evaluator.get_vlm_response_rag(metadata)
            ground_truth = evaluator.load_ground_truth(str(json_dir / f"{image_path.stem}.json"))
            result = evaluator.evaluate_response_llm_judge(vlm_response, ground_truth)
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
    print("\n=== EVALUATION RESULTS ===")
    print(f"Total Images Evaluated: {len(results)}")
    print(f"Average Accuracy: {avg_accuracy:.2%}")
    
    print("\n=== PER-FIELD ACCURACY ===")
    for field, scores in field_accuracies.items():
        field_acc = sum(scores) / len(scores)
        print(f"{field}: {field_acc:.2%}")
    
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
    
    if error_summary:
        df = pd.DataFrame(error_summary)
        print("\n=== ERROR SUMMARY STATISTICS ===")
        print("\nErrors per Field:")
        print(df['Field'].value_counts())
        
        # Save error analysis to CSV
        output_path = "error_analysis_500.csv"
        df.to_csv(output_path, index=False)
        print(f"\nDetailed error analysis saved to: {output_path}")

        print("\n=== EVALUATION RESULTS ===")
        print(f"Total Images Evaluated: {len(results)}")
        print(f"Average Accuracy: {avg_accuracy:.2%}")
        
        print("\n=== PER-FIELD ACCURACY ===")
        for field, scores in field_accuracies.items():
            field_acc = sum(scores) / len(scores)
            print(f"{field}: {field_acc:.2%}")

if __name__ == "__main__":
    run_batch_evaluation(
        image_dir="../data_v2/images_handheld",
        json_dir="../data_v2/json",
        max_samples=500
    )