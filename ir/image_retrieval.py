import os
import json
import torch
import numpy as np
import base64
from PIL import Image
from pydantic import BaseModel, Field
from typing import Dict, Any, Tuple, Optional, Literal
from torchvision import models, transforms
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from together import Together

class ImageRetrieval:
    def __init__(
        self, 
        dataset_dir: str, 
        json_dir: str,
        together_client: Together,
        embedding_type: Literal["ResNet", "ColPali"] = "ResNet",
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
        else:  # ColPali
            self.model, self.processor = self.load_colpali_model()
            
        self.image_embeddings = self.get_or_create_image_embeddings()
        self.neighbors = self.build_knn_index()
        
    def load_resnet50_model(self):
        model = models.resnet50(pretrained=True)
        model.eval()
        return model.to(self.device)
    
    def load_colpali_model(self) -> Tuple[ColPali, ColPaliProcessor]:
        model_name = "vidore/colpali-v1.2"
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        
        processor = ColPaliProcessor.from_pretrained(model_name)
        if not isinstance(processor, BaseVisualRetrieverProcessor):
            raise ValueError("Processor should be a BaseVisualRetrieverProcessor")
            
        return model, processor
    
    def get_or_create_image_embeddings(self):
        if os.path.exists(self.vector_db_path):
            print(f"Loading {self.embedding_type} embeddings from {self.vector_db_path}...")
            return np.load(self.vector_db_path)
        else:
            print(f"Generating {self.embedding_type} embeddings and saving to {self.vector_db_path}...")
            embeddings = self.extract_image_embeddings()
            np.save(self.vector_db_path, embeddings)
            return embeddings

    def extract_image_embeddings(self):
        embeddings = []
        image_files = [f for f in os.listdir(self.dataset_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if self.embedding_type == "ResNet":
            for filename in tqdm(image_files):
                image_path = os.path.join(self.dataset_dir, filename)
                image = self.preprocess_image_resnet(image_path)
                with torch.no_grad():
                    embedding = self.model(image.unsqueeze(0).to(self.device))
                    embedding = embedding.squeeze().cpu().numpy()
                embeddings.append(embedding)
                
        else:  # ColPali
            batch_size = 4
            for i in tqdm(range(0, len(image_files), batch_size)):
                batch_files = image_files[i:i + batch_size]
                batch_images = [Image.open(os.path.join(self.dataset_dir, f)).convert("RGB") 
                              for f in batch_files]
                
                # Process batch using ColPali processor
                batch_inputs = self.processor.process_images(batch_images)
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                
                with torch.no_grad():
                    batch_embeddings = self.model(**batch_inputs)
                    batch_embeddings = batch_embeddings.cpu()
                
                embeddings.extend(torch.unbind(batch_embeddings))
        
        return np.array([emb.numpy() if torch.is_tensor(emb) else emb for emb in embeddings])

    def preprocess_image_resnet(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            )
        ])
        return transform(image)

    def build_knn_index(self):
        neighbors = NearestNeighbors(n_neighbors=self.k, metric="cosine")
        neighbors.fit(self.image_embeddings)
        return neighbors

    def retrieve_similar_images(
        self, 
        image_path: str
    ) -> list[tuple[str, Dict[str, Any], float]]:
        """
        Retrieve k most similar images and their metadata.
        
        Returns:
        List of tuples containing (image_path, metadata, similarity_score)
        """
        if self.embedding_type == "ResNet":
            image = self.preprocess_image_resnet(image_path)
            with torch.no_grad():
                query_embedding = self.model(
                    image.unsqueeze(0).to(self.device)
                ).squeeze().cpu().numpy()
        else:  # ColPali
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor.process_images([image])
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                query_embedding = self.model(**inputs).cpu().numpy()

        distances, indices = self.neighbors.kneighbors(
            [query_embedding] if self.embedding_type == "ResNet" else query_embedding,
            n_neighbors=self.k
        )
        
        results = []
        image_files = sorted([f for f in os.listdir(self.dataset_dir) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        for idx, dist in zip(indices[0], distances[0]):
            similar_image_path = os.path.join(self.dataset_dir, image_files[idx])
            metadata = self.load_image_metadata(similar_image_path)
            similarity_score = 1 - dist  # Convert distance to similarity score
            results.append((similar_image_path, metadata, similarity_score))
            
        return results

    def preprocess_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        return transform(image)
    
    def retrieve_most_similar_image(self, image_path: str):
        print("\nComparing image to WikiArt corpus...")
        image = self.preprocess_image(image_path)
        with torch.no_grad():
            query_embedding = self.model(image.unsqueeze(0).to("cpu")).squeeze().cpu().numpy()
        distances, indices = self.neighbors.kneighbors([query_embedding], self.k)
        most_similar_image_path = os.path.join(self.dataset_dir, os.listdir(self.dataset_dir)[indices[0][0]])
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            similar_image_path = os.path.join(self.dataset_dir, os.listdir(self.dataset_dir)[idx])
            metadata = self.load_image_metadata(similar_image_path)
            similarity_score = 1 - dist  # Convert distance to similarity score
            results.append((similar_image_path, metadata, similarity_score))
        
        if all([1-x < self.sim_threshold for x in distances[0]]):
            print(f"No exact match found. Top similarity score: {results[0][2]}")
            estimate = {}
            if self.top_k_averaging:
                print("\nEstimating from top k results...")
                # Get top k results
                top_k_results = results[:self.k]
                
                # Count occurrences of each attribute
                artist_counts = {}
                style_counts = {}
                
                # Collect counts for each attribute
                for _, metadata, _ in top_k_results:
                    # Count artists
                    if 'artist' in metadata:
                        artist = metadata['artist']
                        artist_counts[artist] = artist_counts.get(artist, 0) + 1
                    
                    # Count styles
                    if 'style' in metadata:
                        style = metadata['style']
                        style_counts[style] = style_counts.get(style, 0) + 1
                
                # Get majority values (or first occurrence in case of ties)
                majority_artist = max(artist_counts.items(), key=lambda x: x[1])[0]
                majority_style = max(style_counts.items(), key=lambda x: x[1])[0]

                print(f"Similar artists: {artist_counts.keys()}")
                print(f"Similar styles: {style_counts.keys()}")
                
                # Create aggregated metadata
                estimate = {
                    'artist': majority_artist,
                    'style': majority_style,
                }
            else:
                estimate = self.vlm_estimate(image_path)

            # Return top image path with estimated metadata and its similarity score
            print(f"Estimated ground truth data: \n{estimate}")
            return (results[0][0], estimate, results[0][2])

        print(f"Exact match found: \n{results[0][0]}")
        print(f"Scraped ground truth data: \n{results[0][1]}")
        return results[0]

    def vlm_estimate(self, image_path):
        print("\nEstimating artist/style from VLM call...")
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
            raise Exception("Error decoding LLm response: ", e)
        
    def load_image_metadata(self, image_path: str) -> Dict[str, Any]:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(self.json_dir, f"{filename}.json")
        with open(json_path, "r") as f:
            metadata = json.load(f)
        # Remove unused location key
        if "location" in metadata:
            del metadata["location"]
        return metadata

# Usage example
def main():
    # Initialize with ResNet embeddings
    resnet_retriever = ImageRetrieval(
        dataset_dir="./data_v3/images",
        json_dir="./data_v3/json",
        embedding_type="ResNet"
    )
    
    # # Initialize with ColPali embeddings
    # colpali_retriever = ImageRetrieval(
    #     dataset_dir="../data_v2/images",
    #     json_dir="../data_v2/json",
    #     embedding_type="ColPali"
    # )
    
    # Query image path
    query_image = "./ir/fenetre-ouverte.jpg"
    # Get similar images using both methods
    print("\nResNet Results:")
    resnet_results = resnet_retriever.retrieve_most_similar_image(query_image)
    
    print(f"Image: {resnet_results[0]}")
    print(f"Similarity Score: {resnet_results[2]:.3f}")
    print(f"Metadata: {resnet_results[1]}\n")
    
    # print("\nColPali Results:")
    # colpali_results = colpali_retriever.retrieve_similar_images(query_image, k=3)
    # for path, metadata, score in colpali_results:
    #     print(f"Image: {path}")
    #     print(f"Similarity Score: {score:.3f}")
    #     print(f"Metadata: {metadata}\n")

if __name__ == "__main__":
    main()