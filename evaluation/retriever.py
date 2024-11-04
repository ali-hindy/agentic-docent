import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, Any, Optional
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class ImageRetrieval:
    def __init__(self, dataset_dir: str, json_dir: str, vector_db_path: str = "vector_database.npy"):
        self.dataset_dir = dataset_dir
        self.json_dir = json_dir
        self.vector_db_path = vector_db_path
        self.model = self.load_resnet50_model()
        self.image_embeddings = self.get_or_create_image_embeddings()
        self.neighbors = self.build_knn_index()

    def load_resnet50_model(self):
        model = models.resnet50(pretrained=True)
        model.eval()
        return model.to("cpu")

    def get_or_create_image_embeddings(self):
        if os.path.exists(self.vector_db_path):
            print(f"Loading image embeddings from {self.vector_db_path}...")
            return np.load(self.vector_db_path)
        else:
            print(f"Generating image embeddings and saving to {self.vector_db_path}...")
            embeddings = self.extract_image_embeddings()
            np.save(self.vector_db_path, embeddings)
            return embeddings

    def extract_image_embeddings(self):
        embeddings = []
        for filename in tqdm(os.listdir(self.dataset_dir)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(self.dataset_dir, filename)
                image = self.preprocess_image(image_path)
                with torch.no_grad():
                    embedding = self.model(image.unsqueeze(0).to("cpu")).squeeze().cpu().numpy()
                embeddings.append(embedding)
        return np.array(embeddings)

    def preprocess_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        return transform(image)

    def build_knn_index(self):
        neighbors = NearestNeighbors(n_neighbors=1, metric="cosine")
        neighbors.fit(self.image_embeddings)
        return neighbors

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
image_retriever = ImageRetrieval("../data_v2/images", "../data_v2/json")
similar_image_path, metadata = image_retriever.retrieve_most_similar_image("../data_v2/images/adam-baltatu_still-life-with-travel-props.jpg")
print(f"Most similar image: {similar_image_path}")
print(f"Metadata: {metadata}")
