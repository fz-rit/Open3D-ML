
"""
Calculate weight (frequency) for each class in the Francisco Forest Segmentation Synthetic dataset.
The class weights are calculated based on the frequency of each class in the dataset.
The class weights are used to balance the classes during training.

Get the files in the directory of the dataset.
For each file in the dataset:
    read the file
    get the class labels
    count the frequency of each class
    add the frequency to the total frequency of each class

Print the total frequency of each class.
"""


from tqdm import tqdm
from pathlib import Path

def calculate_class_frequencies_and_weights(
    data_dir: Path
):
    """
    Scans all .txt files in data_dir, reads the last column as the class label,
    accumulates total frequency per class, and 
    [optional] computes class weights based on inverse frequency.
    """

    # Dictionary to hold total frequency per class
    class_counts = {}

    # Gather all .txt files in the specified directory
    txt_files = list(data_dir.glob("*.txt"))

    # Use tqdm to track progress over the list of files
    for txt_file in tqdm(txt_files, desc="Processing files"):
        with open(txt_file, "r") as f:
            for line in f:
                # Each row is "x y z label" separated by spaces
                # We assume label is in the last column
                parts = line.strip().split()
                if not parts:
                    continue  # skip empty lines
                # Convert label to int (handle float label if needed)
                label = int(float(parts[-1]))
                class_counts[label] = class_counts.get(label, 0) + 1

    # Print the raw frequencies
    print("\nTotal frequency (count) for each class:")
    for cls, freq in sorted(class_counts.items()):
        print(f"Class {cls}: {freq}")

    # # Compute and print class weights as inverse frequency (you can tweak the formula if needed)
    # total_points = sum(class_counts.values())
    # class_weights = {}
    # for cls, freq in class_counts.items():
    #     class_weights[cls] = total_points / float(freq)

    # print("\nClass weights (based on inverse frequency):")
    # for cls, weight in sorted(class_weights.items()):
    #     print(f"Class {cls}: {weight:.4f}")


if __name__ == "__main__":
    data_dir = Path("/home/fzhcis/mylab/data/synthetic-lidar-point-clouds-tree-simulator/LiDAR-like-Dataset/dataForest/12345678")
    calculate_class_frequencies_and_weights(data_dir)

