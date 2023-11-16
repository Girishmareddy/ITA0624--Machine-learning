import pandas as pd

# Sample dataset
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny', 'Sunny'],
    'Temperature': ['Warm', 'Warm', 'Cold', 'Warm', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High', 'Normal'],
    'Wind': ['Strong', 'Strong', 'Weak', 'Strong', 'Weak'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

def find_s_algorithm(dataset):
    # Initialize the most specific hypothesis
    hypothesis = dataset.iloc[0, :-1].to_dict()

    # Iterate through the dataset
    for i in range(1, len(dataset)):
        instance = dataset.iloc[i, :-1].to_dict()
        target = dataset.iloc[i, -1]

        # Check if the instance is positive (EnjoySport is 'Yes')
        if target == 'Yes':
            for attr, value in instance.items():
                # Update the hypothesis if the attribute value is different
                if value != hypothesis[attr]:
                    hypothesis[attr] = '?'

    return hypothesis

# Apply the Find-S algorithm to the dataset
specific_hypothesis = find_s_algorithm(df)

# Display the most specific hypothesis
print("Most Specific Hypothesis:")
print(specific_hypothesis)
