import numpy as np


if __name__ == "__main__":
    # Generate random numbers and save them to a file
    np.random.seed(42)  # Set the seed for reproducibility
    version = "v2" # ["v1", "v2"]
    if version == "v1":
        num_choices = 3
        save_path = "random_numbers.npy"
    elif version == "v2":
        num_choices = 45 # 3 or 45
        save_path = "random_numbers_v2.npy"
    else:
        raise ValueError("Invalid version")
    random_numbers = np.random.choice(list(range(num_choices)), size=500000)
    np.save(save_path, random_numbers)
    loaded_random_numbers = np.load(save_path)
    print(loaded_random_numbers[:100])