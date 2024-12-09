import pprint
from typing import List, cast
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device


def get_image_embeddings(
    model: ColPali,
    processor: ColPaliProcessor,
    images: List,
    batch_size: int = 4,
    device: str = "cuda",
) -> List[torch.Tensor]:
    """
    Get embeddings for a list of images.
    """
    dataloader = DataLoader(
        dataset=ListDataset[str](images),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )

    embeddings: List[torch.Tensor] = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_embeddings = model(**batch)
        embeddings.extend(list(torch.unbind(batch_embeddings.to("cpu"))))

    return embeddings


def main():
    """
    Example script to run image-to-image retrieval with ColPali.
    """
    device = get_torch_device("auto")
    print(f"Device used: {device}")

    # Model name
    model_name = "vidore/colpali-v1.2"

    # Load model
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    # Load processor
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))
    if not isinstance(processor, BaseVisualRetrieverProcessor):
        raise ValueError("Processor should be a BaseVisualRetrieverProcessor")

    # Load dataset
    dataset = cast(
        Dataset, load_dataset("vidore/docvqa_test_subsampled", split="test[:16]")
    )
    database_images = dataset["image"]

    # Select query images (for demonstration, using some images from the dataset)
    query_indices = [12, 15]
    query_images = [dataset[idx]["image"] for idx in query_indices]

    print("Processing database images...")
    database_embeddings = get_image_embeddings(
        model, processor, database_images, device=device
    )

    print("Processing query images...")
    query_embeddings = get_image_embeddings(
        model, processor, query_images, device=device
    )

    # Run scoring
    scores = processor.score(query_embeddings, database_embeddings).cpu().numpy()

    # Get top-k results for each query
    k = 5  # Number of similar images to retrieve
    top_k_indices = scores.argsort(axis=1)[:, -k:][:, ::-1]

    # Print results
    print("\nTop-k similar images for each query:")
    for query_idx, retrieved_indices in enumerate(top_k_indices):
        print(f"\nQuery image index: {query_indices[query_idx]}")
        print(f"Top {k} similar images (indices): {retrieved_indices.tolist()}")
        print(f"Similarity scores: {scores[query_idx][retrieved_indices].tolist()}")


if __name__ == "__main__":
    main()
