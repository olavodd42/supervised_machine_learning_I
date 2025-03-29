import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the breast cancer dataset
breast_cancer_data = load_breast_cancer()

# Explore the data
print('Data:')
print(breast_cancer_data.data[0])
print('Features:')
print(breast_cancer_data.feature_names)
print('Target data:')
print(breast_cancer_data.target)
print('Target names:')
print(breast_cancer_data.target_names)

# Split data into training and validation sets
training_data, validation_data, training_labels, validation_labels = train_test_split(
    breast_cancer_data.data, 
    breast_cancer_data.target, 
    test_size=0.2, 
    random_state=100
)

# Confirm data split worked correctly
print(f'Training data length: {len(training_data)}')
print(f'Training labels length: {len(training_labels)}')

# Create lists to store k values and corresponding accuracies
k_list = list(range(1, 101))
accuracies = []

# Test different k values
for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracy = classifier.score(validation_data, validation_labels)
    accuracies.append(accuracy)

# Plot the results
plt.plot(k_list, accuracies)
plt.xlabel("k")  # Exactly as specified in the instructions
plt.ylabel("Validation Accuracy")  # Exactly as specified in the instructions
plt.title("Breast Cancer Classifier Accuracy")
plt.show()