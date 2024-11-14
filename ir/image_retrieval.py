import os
import json
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional, Literal
from torchvision import models, transforms
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

class ImageRetrieval:
    def __init__(
        self, 
        dataset_dir: str, 
        json_dir: str, 
        embedding_type: Literal["ResNet", "ColPali"] = "ResNet",
        vector_db_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.dataset_dir = dataset_dir
        self.json_dir = json_dir
        self.embedding_type = embedding_type
        self.device = device
        
        # Set vector database path
        if vector_db_path is None:
            self.vector_db_path = f"vector_database_{embedding_type.lower()}.npy"
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
        neighbors = NearestNeighbors(n_neighbors=5, metric="cosine")
        neighbors.fit(self.image_embeddings)
        return neighbors

    def retrieve_similar_images(
        self, 
        image_path: str, 
        k: int = 5
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
            n_neighbors=k
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
        image = self.preprocess_image(image_path)
        with torch.no_grad():
            query_embedding = self.model(image.unsqueeze(0).to("cpu")).squeeze().cpu().numpy()
        distances, indices = self.neighbors.kneighbors([query_embedding])
        most_similar_image_path = os.path.join(self.dataset_dir, os.listdir(self.dataset_dir)[indices[0][0]])
        return most_similar_image_path, self.load_image_metadata(most_similar_image_path)

    def load_image_metadata(self, image_path: str) -> Dict[str, Any]:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(self.json_dir, f"{filename}.json")
        with open(json_path, "r") as f:
            metadata = json.load(f)
        return metadata

# Usage example
def main():
    # Initialize with ResNet embeddings
    resnet_retriever = ImageRetrieval(
        dataset_dir="../data_v2/images",
        json_dir="../data_v2/json",
        embedding_type="ResNet"
    )
    
    # Initialize with ColPali embeddings
    colpali_retriever = ImageRetrieval(
        dataset_dir="../data_v2/images",
        json_dir="../data_v2/json",
        embedding_type="ColPali"
    )
    
    # Query image path
    query_image = "../data_v2/images/adam-baltatu_still-life-with-travel-props.jpg"
    
    # Get similar images using both methods
    print("\nResNet Results:")
    resnet_results = resnet_retriever.retrieve_similar_images(query_image, k=3)
    for path, metadata, score in resnet_results:
        print(f"Image: {path}")
        print(f"Similarity Score: {score:.3f}")
        print(f"Metadata: {metadata}\n")
    
    print("\nColPali Results:")
    colpali_results = colpali_retriever.retrieve_similar_images(query_image, k=3)
    for path, metadata, score in colpali_results:
        print(f"Image: {path}")
        print(f"Similarity Score: {score:.3f}")
        print(f"Metadata: {metadata}\n")

if __name__ == "__main__":
    main()