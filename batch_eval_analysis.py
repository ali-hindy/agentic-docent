from pathlib import Path
from eval import ArtEvaluator
from pipeline import DocentPipeline
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import os 

def run_batch_evaluation(image_dir: str, json_dir: str, max_samples: int = 5) -> None:
    """
    Performs batch evaluation comparing DocentPipeline against Llama-3.2 baseline.
    Evaluates both factual accuracy and stylistic qualities for both approaches.
    
    Args:
        image_dir (str): Path to directory containing artwork images
        json_dir (str): Path to directory containing ground truth JSON files
        max_samples (int, optional): Maximum number of images to evaluate. Default: 5
    
    Returns:
        None. Results are printed to console and saved to files:
            - docent_factual_error_analysis.csv: Factual errors for DocentPipeline
            - docent_style_analysis.csv: Style analysis for DocentPipeline
            - baseline_factual_error_analysis.csv: Factual errors for Llama baseline
            - baseline_style_analysis.csv: Style analysis for Llama baseline
    """
    evaluator = ArtEvaluator()
    docent = DocentPipeline(image_dir, json_dir, os.getenv('TOGETHER_API_KEY'), embedding_type='CLIP')
    image_dir = Path(image_dir)
    json_dir = Path(json_dir)
    
    # Store results for both approaches
    results = {
        'docent': {
            'factual': {},
            'style': {},
            'error_analysis': defaultdict(list),
            'style_analysis': defaultdict(list)
        },
        'baseline': {
            'factual': {},
            'style': {},
            'error_analysis': defaultdict(list),
            'style_analysis': defaultdict(list)
        }
    }
    
    # Track progress
    cnt = 0
    for image_path in tqdm(image_dir.glob("*.jpg")):
        if cnt >= max_samples:
            break
            
        gt_path = json_dir / f"{image_path.stem}.json"
        if gt_path.exists():
            try:
                ground_truth = evaluator.load_ground_truth(str(gt_path))
                
                # Evaluate both approaches
                for approach in ['docent', 'baseline']:
                    # Get response based on approach
                    if approach == 'docent':
                        response = docent.run(str(image_path))
                        print(response)
                    else:
                        response = evaluator.get_baseline_full_response(str(image_path))
                    
                    # Evaluate factual accuracy
                    factual_result = evaluator.evaluate_text_response_llm_judge(
                        response, ground_truth
                    )
                    results[approach]['factual'][image_path.name] = factual_result
                    
                    # Evaluate style
                    style_result = evaluator.evaluate_style_response_llm_judge(response)
                    results[approach]['style'][image_path.name] = style_result
                    
                    # Collect factual errors
                    for field, score in factual_result['field_scores'].items():
                        if score == 0:
                            if field in factual_result['incorrect_fields']:
                                extracted = factual_result['incorrect_fields'][field]['extracted']
                                ground_truth_value = factual_result['incorrect_fields'][field]['ground_truth']
                            else:
                                extracted = factual_result['extracted_values'].get(field, 'MISSING')
                                ground_truth_value = ground_truth.get(field, 'MISSING')
                            
                            results[approach]['error_analysis'][field].append({
                                'image': image_path.name,
                                'response': extracted,
                                'ground_truth': ground_truth_value,
                                'present_in_text': factual_result['incorrect_fields'].get(field, {}).get('present_in_text', False)
                            })
                    
                    # Collect style analysis
                    for criterion, score in style_result['style_scores'].items():
                        results[approach]['style_analysis'][criterion].append({
                            'image': image_path.name,
                            'score': score,
                            'evidence': style_result['style_feedback'][criterion]
                        })
                
                cnt += 1
                
            except Exception as e:
                print(f"Error processing {image_path.name}: {str(e)}")
                continue
    
    # Print and save results for both approaches
    for approach in ['docent', 'baseline']:
        print(f"\n=== {approach.upper()} RESULTS ===")
        
        # Calculate factual accuracy
        factual_acc = sum(val["accuracy"] for val in results[approach]['factual'].values())
        avg_factual_accuracy = factual_acc / len(results[approach]['factual']) if results[approach]['factual'] else 0
        
        # Calculate style scores
        style_acc = sum(val["overall_style_score"] for val in results[approach]['style'].values())
        avg_style_score = style_acc / len(results[approach]['style']) if results[approach]['style'] else 0
        
        print(f"\nFactual Accuracy: {avg_factual_accuracy:.2%}")
        print(f"Style Score: {avg_style_score:.2%}")
        
        # Per-field factual accuracy
        print("\nPer-field Accuracy:")
        field_accuracies = defaultdict(list)
        for result in results[approach]['factual'].values():
            for field, score in result['field_scores'].items():
                field_accuracies[field].append(score)
        
        for field, scores in field_accuracies.items():
            field_acc = sum(scores) / len(scores)
            print(f"{field}: {field_acc:.2%}")
        
        # Per-criterion style scores
        print("\nPer-criterion Style Scores:")
        for criterion, analyses in results[approach]['style_analysis'].items():
            criterion_avg = sum(analysis['score'] for analysis in analyses) / len(analyses)
            print(f"{criterion}: {criterion_avg:.2%}")
        
        # Save error analysis
        error_df = pd.DataFrame([
            {
                'Field': field,
                'Image': error['image'],
                'Response': error['response'],
                'Ground Truth': error['ground_truth']
            }
            for field, errors in results[approach]['error_analysis'].items()
            for error in errors
        ])
        if not error_df.empty:
            error_path = f"{approach}_factual_error_analysis.csv"
            error_df.to_csv(error_path, index=False)
            print(f"\nFactual error analysis saved to: {error_path}")
        
        # Save style analysis
        style_df = pd.DataFrame([
            {
                'Criterion': criterion,
                'Image': analysis['image'],
                'Score': analysis['score'],
                'Evidence': analysis['evidence']
            }
            for criterion, analyses in results[approach]['style_analysis'].items()
            for analysis in analyses
        ])
        if not style_df.empty:
            style_path = f"{approach}_style_analysis.csv"
            style_df.to_csv(style_path, index=False)
            print(f"\nStyle analysis saved to: {style_path}")

if __name__ == "__main__":
    run_batch_evaluation(
        image_dir="./data_v2/images_handheld",
        json_dir="./data_v2/json",
        max_samples=5
    )