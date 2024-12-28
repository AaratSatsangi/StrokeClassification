import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from collections import defaultdict

# Assuming you already have your dataset and sampler set up
dataset = ImageFolder(root='path_to_your_dataset')
class_weights = [1.0] * len(dataset.classes)  # Example weights
sample_weights = [class_weights[label] for _, label in dataset.samples]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)

# DataLoader with the WeightedRandomSampler
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# Initialize counters
image_seen_count = defaultdict(int)  # Tracks how many times each image is seen
class_count_per_batch = []  # Stores class distribution per batch

# Iterate through the DataLoader
for epoch in range(1):  # Example for one epoch
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Update image seen count
        for idx in dataloader.sampler:
            image_seen_count[idx] += 1

        # Count occurrences of each class in the current batch
        batch_class_count = defaultdict(int)
        for label in labels.numpy():
            batch_class_count[label] += 1
        class_count_per_batch.append(batch_class_count)

# Display results
print("Image seen count during epoch:")
for idx, count in image_seen_count.items():
    print(f"Image {idx}: {count} times")

print("\nClass count per batch:")
for batch_idx, batch_class_count in enumerate(class_count_per_batch):
    print(f"Batch {batch_idx}: {dict(batch_class_count)}")
