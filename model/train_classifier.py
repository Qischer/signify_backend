import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Separate data and labels
data = data_dict['data']
labels = data_dict['labels']

# Identify the number of samples with length 84
lengths = [len(sample) for sample in data]
num_samples_length_84 = lengths.count(84)
print(f"Number of samples with length 84: {num_samples_length_84}")

# Filter out samples with length not equal to 42
filtered_data = [sample for sample in data if len(sample) == 42]
filtered_labels = [label for sample, label in zip(data, labels) if len(sample) == 42]

# Convert to NumPy arrays
data_array = np.array(filtered_data)
labels_array = np.array(filtered_labels)

# Proceed with training
x_train, x_test, y_train, y_test = train_test_split(
    data_array, labels_array, test_size=0.2, shuffle=True, stratify=labels_array
)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
