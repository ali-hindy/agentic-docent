# Art Docent Evaluator

A robust Python script for evaluating Vision Language Model (VLM) performance on art analysis tasks. This tool compares VLM-generated artwork descriptions against ground truth data and provides detailed accuracy metrics.

## Features

- Automated evaluation of VLM responses against ground truth data
- Support for image-based art analysis using the Together API
- Detailed accuracy metrics and comparison reporting
- Comprehensive error handling and logging
- Type hints for better code maintainability

## Installation

1. Clone the repository:
```bash
git clone git@github.com:ali-hindy/agentic-docent.git
cd art-evaluator
```

2. Change to evaluation directory
```bash
cd evaluation 
```

2. Install required dependencies:
```bash
pip install together pillow
```

## Configuration

1. Set up your Together API key:
   - Option 1: Set environment variable:
     ```bash
     export TOGETHER_API_KEY=your_api_key_here
     ```
   - Option 2: Pass directly to the ArtEvaluator class:
     ```python
     evaluator = ArtEvaluator(api_key="your_api_key_here")
     ```

2. Prepare your input files:
   - Place your test image as `input_image.png` in the script directory
   - Create a `ground_truth.json` file with the correct artwork information

### Ground Truth JSON Format

Your `ground_truth.json` should follow this format:
```json
{
    "artist": "Vincent van Gogh", # full name of artist
    "title_of_work": "The Starry Night", # full title of work
    "date_created": "1889", # YEAR of creation
    "location": "Saint-Rémy-de-Provence, France", # location of creation in (city, country) format
    "style": "Post-Impressionism" # full name of art style
}
```

## Usage

### Basic Usage

Run the script directly:
```bash
python art_evaluator.py
```

### Using as a Module

```python
from art_evaluator import ArtEvaluator

# Initialize evaluator
evaluator = ArtEvaluator()

# Process single image
vlm_response = evaluator.get_vlm_response("path/to/image.png")
ground_truth = evaluator.load_ground_truth("path/to/ground_truth.json")
results = evaluator.evaluate_response(vlm_response, ground_truth)

# Print results
print(f"Accuracy: {results['accuracy']*100:.2f}%")
```

## Output Format

The script provides detailed evaluation results:

```
Evaluation Results:
Accuracy: 80.00%

Correct Matches:
- artist: vincent van gogh
- title_of_work: the starry night
- style: post-impressionism
- date_created: 1889

Mismatches:
- location:
  VLM: saint remy, france
  Ground Truth: saint-rémy-de-provence, france
```

## Error Handling

The script includes comprehensive error handling for common issues:
- Missing input files
- Invalid JSON format
- API connection issues
- Image encoding problems

All errors are logged with detailed messages to help with debugging.

## Logging

Logs are written to console with timestamp and log level:
```
2024-10-23 10:15:30 - INFO - Getting VLM response...
2024-10-23 10:15:32 - INFO - Loading ground truth...
2024-10-23 10:15:32 - INFO - Evaluating response...
```

## Advanced Usage

### Custom Field Comparison

To modify how fields are compared, subclass `ArtEvaluator`:
We aim to add support for LLM-as-a-judge evaluation soon. 
```python
class CustomArtEvaluator(ArtEvaluator):
    def evaluate_response(self, vlm_response, ground_truth):
        # Custom comparison logic
        pass
```

### Batch Processing

To process multiple images:

```python
from pathlib import Path

evaluator = ArtEvaluator()
image_dir = Path("images")
results = {}

for image_path in image_dir.glob("*.png"):
    gt_path = Path("ground_truth") / f"{image_path.stem}.json"
    if gt_path.exists():
        vlm_response = evaluator.get_vlm_response(str(image_path))
        ground_truth = evaluator.load_ground_truth(str(gt_path))
        results[image_path.name] = evaluator.evaluate_response(
            vlm_response, ground_truth
        )
```


## Troubleshooting

### Common Issues

1. **API Key Issues**
   ```
   Error: Together API key not found
   Solution: Ensure your API key is properly set
   ```

2. **File Not Found**
   ```
   Error: Image file not found: input_image.png
   Solution: Ensure input files exist in the correct location
   ```

3. **JSON Parse Errors**
   ```
   Error: Invalid JSON in VLM response
   Solution: Check VLM response format and handling
   ```

### Getting Help

For issues and support:
1. Check the error logs
2. Review the troubleshooting section
3. Open an issue on the repository

## Version History

- 1.0.0
  - Initial release
  - Basic evaluation functionality
  - Together API integration

- To Dos:
  - Model customization as a class parameter
  - LLM-as-a-judge evaluation
  - Batch processing for multiple images & ground truths