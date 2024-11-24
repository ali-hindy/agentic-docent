import base64
import json
import os
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from together import Together
from torchvision import models, transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class ImageRetrieval:
    def __init__(
        self, 
        dataset_dir: str, 
        json_dir: str,
        together_client: Together,
        embedding_type: Literal["ResNet", "CLIP"] = "ResNet",
        vector_db_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sim_threshold: float = 0.9,
        k: int = 5,
        top_k_averaging: bool = False
    ):
        self.dataset_dir = dataset_dir
        self.json_dir = json_dir
        self.together_client = together_client
        self.embedding_type = embedding_type
        self.device = device
        self.sim_threshold = sim_threshold
        self.k = k
        self.top_k_averaging = top_k_averaging

        # Set vector database path
        data_name = dataset_dir.split("/")[1]
        if vector_db_path is None:
            self.vector_db_path = f"vector_database_{embedding_type.lower()}_{data_name}.npy"
        else:
            self.vector_db_path = vector_db_path
            
        # Load appropriate model and processor
        if embedding_type == "ResNet":
            self.model = self.load_resnet50_model()
            self.processor = None
        else:  # CLIP
            self.model, self.processor = self.load_clip_model()
            
        self.image_embeddings = self.get_or_create_image_embeddings()

    def load_resnet50_model(self):
        model = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.embedding_size = model.fc.in_features
        model.fc = torch.nn.Identity()
        model = model.eval().to(self.device)
        return model
    
    def load_clip_model(self) -> Tuple[CLIPModel, CLIPProcessor]:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        model.eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    
    def get_or_create_image_embeddings(self):
        if os.path.exists(self.vector_db_path):
            print(f"Loading {self.embedding_type} embeddings from {self.vector_db_path}...")
            embeddings = np.load(self.vector_db_path)
            # Ensure loaded embeddings are normalized
            return self.normalize_embedding(embeddings)
        else:
            print(f"Generating {self.embedding_type} embeddings and saving to {self.vector_db_path}...")
            embeddings = self.extract_image_embeddings()
            # Normalize before saving
            normalized_embeddings = self.normalize_embedding(embeddings)
            np.save(self.vector_db_path, normalized_embeddings)
            return normalized_embeddings

    def preprocess_image_resnet(self, image_path: str):
        """Consistent preprocessing for ResNet"""
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(256),  # Resize the shorter side to 256
            transforms.CenterCrop(224),  # Take center crop
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]     # ImageNet stds
            )
        ])
        return transform(image)

    def extract_image_embeddings(self):
        """Extract embeddings with consistent preprocessing"""
        embeddings = []
        image_files = sorted([f for f in os.listdir(self.dataset_dir) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if self.embedding_type == "ResNet":
            for filename in tqdm(image_files):
                image_path = os.path.join(self.dataset_dir, filename)
                image = self.preprocess_image_resnet(image_path)
                with torch.no_grad():
                    embedding = self.model(image.unsqueeze(0).to(self.device))
                    embedding = embedding.squeeze().cpu().numpy()
                embeddings.append(embedding)
                
        else:  # CLIP
            batch_size = 8
            for i in tqdm(range(0, len(image_files), batch_size)):
                batch_files = image_files[i:i + batch_size]
                batch_images = [Image.open(os.path.join(self.dataset_dir, f)).convert("RGB") 
                            for f in batch_files]
                
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    batch_embeddings = image_features.cpu().numpy()
                embeddings.extend([emb for emb in batch_embeddings])
        
        return np.array(embeddings, dtype=np.float32)

    def build_similarity_index(self):
        """Build similarity index with proper normalization"""
        print("Normalizing embeddings...")
        
        # Normalize each embedding individually
        norms = np.linalg.norm(self.image_embeddings, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1e-8  # Avoid division by zero
        self.normalized_embeddings = self.image_embeddings / norms
        
        # Verify normalization
        check_norms = np.linalg.norm(self.normalized_embeddings, axis=1)
        print(f"Normalized embedding norms: min={check_norms.min():.6f}, max={check_norms.max():.6f}")
        
        return None

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize a single embedding or batch of embeddings"""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)  # Prevent division by zero
        return embedding / norm
    
    def get_query_embedding(self, image_path: str) -> np.ndarray:
        """Helper function to get embedding for a query image"""
        if self.embedding_type == "ResNet":
            image = self.preprocess_image_resnet(image_path)
            with torch.no_grad():
                embedding = self.model(
                    image.unsqueeze(0).to(self.device)
                ).squeeze().cpu().numpy()
        else:  # CLIP
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=[image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding = image_features.cpu().numpy()
        
        return embedding.astype(np.float32)

    def retrieve_similar_images(
        self, 
        image_path: str
    ) -> list[tuple[str, Dict[str, Any], float]]:
        """Retrieve similar images using consistent preprocessing and normalization"""
        try:
            # Get query embedding using same preprocessing
            if self.embedding_type == "ResNet":
                image = self.preprocess_image_resnet(image_path)
                with torch.no_grad():
                    query_embedding = self.model(image.unsqueeze(0).to(self.device))
                    query_embedding = query_embedding.squeeze().cpu().numpy()
            else:  # CLIP
                image = Image.open(image_path).convert("RGB")
                inputs = self.processor(images=[image], return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    query_embedding = self.model.get_image_features(**inputs)
                    query_embedding = query_embedding.cpu().numpy().squeeze()

            # Normalize query embedding
            query_embedding = self.normalize_embedding(query_embedding).squeeze()
            
            # Get sorted image files (same order as during database creation)
            image_files = sorted([f for f in os.listdir(self.dataset_dir) 
                                if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            # Calculate cosine similarities
            similarities = np.dot(self.image_embeddings, query_embedding)
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:self.k]
            top_scores = similarities[top_indices]
            
            # Create results
            results = []
            for idx, score in zip(top_indices, top_scores):
                similar_image_path = os.path.join(self.dataset_dir, image_files[idx])
                metadata = self.load_image_metadata(similar_image_path)
                results.append((similar_image_path, metadata, float(score)))
            
            return results
    
        except Exception as e:
                print(f"Error in retrieve_similar_images: {str(e)}")
                raise


    def retrieve_most_similar_image(self, image_path: str):
        results = self.retrieve_similar_images(image_path)
        
        if results[0][2] < self.sim_threshold:
            estimate = {}
            if self.top_k_averaging:
                # Get top k results
                top_k_results = results[:self.k]
                
                # Count occurrences of each attribute
                artist_counts = {}
                style_counts = {}
                
                for _, metadata, _ in top_k_results:
                    if 'artist' in metadata:
                        artist = metadata['artist']
                        artist_counts[artist] = artist_counts.get(artist, 0) + 1
                    
                    if 'style' in metadata:
                        style = metadata['style']
                        style_counts[style] = style_counts.get(style, 0) + 1
                
                majority_artist = max(artist_counts.items(), key=lambda x: x[1])[0]
                majority_style = max(style_counts.items(), key=lambda x: x[1])[0]

                print(f"Similar artists: {artist_counts.keys()}")
                print(f"Similar styles: {style_counts.keys()}")
                
                estimate = {
                    'artist': majority_artist,
                    'style': majority_style,
                }
            else:
                estimate = self.vlm_estimate(image_path)

            return (results[0][0], estimate, results[0][2])

        return results[0]

    def vlm_estimate(self, image_path):
        description_prompt = "Identify the artist and artistic style of this artwork. Respond only with valid JSON in the format {\"artist\": \"<ARTIST>\", \"style\": \"<STYLE>\"}. {"
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        response = self.together_client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": description_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            raise Exception("Error decoding LLM response: ", e)
        
    def load_image_metadata(self, image_path: str) -> Dict[str, Any]:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(self.json_dir, f"{filename}.json")
        with open(json_path, "r") as f:
            metadata = json.load(f)
        if "location" in metadata:
            del metadata["location"]
        return metadata

def evaluate_retrieval_accuracy(retriever: ImageRetrieval, test_dir: str) -> dict:
    """
    Evaluates the retrieval accuracy by checking if each image retrieves itself as the most similar.
    
    Args:
        retriever: ImageRetrieval instance
        test_dir: Directory containing test images
        
    Returns:
        Dictionary containing accuracy metrics
    """
    total_images = 0
    correct_retrievals = 0
    top_k_correct = 0
    
    # Get all image files from both directories
    test_files = [f for f in os.listdir(test_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))][:500]
    
    # Get the files in the retriever's database
    db_files = [f for f in os.listdir(retriever.dataset_dir) 
                if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Create a mapping of filenames to their indices in the database
    db_file_to_idx = {fname: idx for idx, fname in enumerate(db_files)}
    
    print(f"\nEvaluating {len(test_files)} images...")
    print(f"Database contains {len(db_files)} images")
    
    results = []
    for test_file in tqdm(test_files):
        if test_file not in db_file_to_idx:
            print(f"Warning: {test_file} not found in database")
            continue
            
        query_path = os.path.join(test_dir, test_file)
        total_images += 1
        
        # Get similar images
        similar_images = retriever.retrieve_similar_images(query_path)
        
        # Get base names for comparison
        query_base = os.path.basename(query_path)
        retrieved_bases = [os.path.basename(path) for path, _, _ in similar_images]
        
        # Get the expected index of the query image in the database
        expected_idx = db_file_to_idx[query_base]
        
        # Check if the most similar image is the query image itself
        if retrieved_bases[0] == query_base:
            correct_retrievals += 1
            
        # Check if the query image is in the top k results
        if query_base in retrieved_bases[:retriever.k]:
            top_k_correct += 1
            
        # Store detailed results
        results.append({
            'query': query_base,
            'top_1_match': retrieved_bases[0],
            'correct': retrieved_bases[0] == query_base,
            'top_k_correct': query_base in retrieved_bases[:retriever.k],
            'similarity_score': similar_images[0][2],
            'all_matches': retrieved_bases[:retriever.k],
            'all_scores': [score for _, _, score in similar_images[:retriever.k]]
        })
        
        # Print details for incorrect retrievals
        if not retrieved_bases[0] == query_base:
            print(f"\nIncorrect retrieval:")
            print(f"Query: {query_base}")
            print(f"Retrieved: {retrieved_bases[0]}")
            print(f"Similarity score: {similar_images[0][2]:.3f}")
            print(f"Top {retriever.k} matches: {retrieved_bases[:retriever.k]}")
            print(f"Top {retriever.k} scores: {[f'{score:.3f}' for _, _, score in similar_images[:retriever.k]]}")
    
    # Calculate metrics
    top_1_accuracy = correct_retrievals / total_images if total_images > 0 else 0
    top_k_accuracy = top_k_correct / total_images if total_images > 0 else 0
    
    # Find failure cases
    failure_cases = [r for r in results if not r['correct']]
    
    # Calculate average similarity score for correct and incorrect retrievals
    correct_scores = [r['similarity_score'] for r in results if r['correct']]
    incorrect_scores = [r['similarity_score'] for r in results if not r['correct']]
    
    avg_correct_score = np.mean(correct_scores) if correct_scores else 0
    avg_incorrect_score = np.mean(incorrect_scores) if incorrect_scores else 0
    
    metrics = {
        'total_images': total_images,
        'top_1_accuracy': top_1_accuracy,
        'top_k_accuracy': top_k_accuracy,
        'correct_retrievals': correct_retrievals,
        'top_k_correct': top_k_correct,
        'avg_correct_similarity': avg_correct_score,
        'avg_incorrect_similarity': avg_incorrect_score,
        'failure_cases': failure_cases[:10]  # Only keep first 10 failures for readability
    }
    
    return metrics

def print_evaluation_results(metrics: dict):
    """Prints the evaluation results in a formatted way."""
    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"Total Images Evaluated: {metrics['total_images']}")
    print(f"Top-1 Accuracy: {metrics['top_1_accuracy']:.2%}")
    print(f"Top-k Accuracy: {metrics['top_k_accuracy']:.2%}")
    print(f"Correct Retrievals: {metrics['correct_retrievals']}")
    print(f"Average Similarity Score (Correct): {metrics['avg_correct_similarity']:.3f}")
    print(f"Average Similarity Score (Incorrect): {metrics['avg_incorrect_similarity']:.3f}")
    
    print("\nSample Failure Cases:")
    print("-" * 50)
    for case in metrics['failure_cases']:
        print(f"Query: {case['query']}")
        print(f"Retrieved: {case['top_1_match']}")
        print(f"Similarity Score: {case['similarity_score']:.3f}")
        print(f"Top matches: {case['all_matches']}")
        print(f"Top scores: {[f'{score:.3f}' for score in case['all_scores']]}")
        print("-" * 30)


# Usage example
def main():
    client = Together()
    # Initialize with ResNet embeddings
    resnet_retriever = ImageRetrieval(
        dataset_dir="./data_v3/images",
        json_dir="./data_v3/json",
        embedding_type="ResNet",
        together_client=client
    )
    
    # Initialize with clip embeddings
    clip_retriever = ImageRetrieval(
        dataset_dir="./data_v3/images",
        json_dir="./data_v3/json",
        embedding_type="CLIP",
        together_client=client
    )
    
    # Query image path
    query_image = "./ir/fenetre-ouverte.jpg"
    # Get similar images using both methods
    print("\nResNet Results:")
    resnet_results = resnet_retriever.retrieve_similar_images(query_image)[0]
    
    print(f"Image: {resnet_results[0]}")
    print(f"Similarity Score: {resnet_results[2]:.3f}")
    print(f"Metadata: {resnet_results[1]}\n")
    
    print("\nColPali Results:")
    clip_results = clip_retriever.retrieve_similar_images(query_image)[0]
    print(f"Image: {clip_results[0]}")
    print(f"Similarity Score: {clip_results[2]:.3f}")
    print(f"Metadata: {clip_results[1]}\n")

    # Batch eval the entire test directory
    # embeddings_to_test = ["ResNet", "CLIP"]
    # test_dir = "./data_v3/images" 
    # for embedding_type in embeddings_to_test:
    #     print(f"\nEvaluating {embedding_type} embeddings...")
    #     print("=" * 50)
        
    #     # Initialize retriever
    #     retriever = ImageRetrieval(
    #         dataset_dir=test_dir,
    #         json_dir="./data_v3/json",
    #         embedding_type=embedding_type,
    #         together_client=client,
    #         k=5  # Adjust k as needed
    #     )
        
    #     # Run evaluation
    #     metrics = evaluate_retrieval_accuracy(retriever, test_dir)
        
    #     # Print results
    #     print_evaluation_results(metrics)
        

if __name__ == "__main__":
    main()