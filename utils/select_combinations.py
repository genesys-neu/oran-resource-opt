import random

# Function to generate valid tuples where no slice has 0 users and the first slice is limited to 3 users
def generate_valid_combinations(min_users, max_users, max_first_slice):
    valid_combinations = []
    for total_users in range(min_users, max_users + 1):
        for a in range(1, min(total_users - 1, max_first_slice + 1)):  # Slice 1, max of 3 users
            for b in range(1, total_users - a):  # Slice 2, at least 1 user
                c = total_users - a - b  # Slice 3, the remainder
                if c >= 1:  # Slice 3 must have at least 1 user
                    valid_combinations.append((a, b, c))
    return valid_combinations

# Generate all valid combinations with user sum between 3 and 7, no slice has 0 users, and the first slice <= 3
valid_combinations = generate_valid_combinations(3, 7, 3)

# Select 15 random combinations from the valid combinations
selected_combinations = random.sample(valid_combinations, 15)

# Print the selected combinations
for idx, combo in enumerate(selected_combinations):
    print(f"Selection {idx + 1}: {combo}")
