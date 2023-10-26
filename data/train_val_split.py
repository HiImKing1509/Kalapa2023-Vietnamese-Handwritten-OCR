import random

# Total number of images
total_images = 2700

# Calculate the number of images for training and testing
train_ratio = 0.8
test_ratio = 0.2
num_train_images = int(total_images * train_ratio)
num_test_images = int(total_images * test_ratio)

# Generate a list of all numbers from 0 to 2699
all_numbers = list(range(total_images))

# Randomly shuffle the list
random.shuffle(all_numbers)

# Split the shuffled list into training and testing data
train_data = all_numbers[:num_train_images]
test_data = all_numbers[num_train_images:num_train_images + num_test_images]

# Function to write the data to a text file


def write_data_to_file(data, filename):
    with open(filename, 'w') as file:
        for number in data:
            file.write(f'{number}.jpg\n')


# Write the training data to a file
write_data_to_file(train_data, 'data/train')

# Write the test data to a file
write_data_to_file(test_data, 'data/test')

print(f"Number of training images: {len(train_data)}")
print(f"Number of testing images: {len(test_data)}")
