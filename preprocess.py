import json
with open('labeled_result_corrected.json', 'r') as file:
    dataset = json.load(file)

data = []
labels = []

for paper_title, paper_data in dataset.items():
    main_text = paper_title + " " + paper_data["abstract"]
    for reference in paper_data["references"]:
        data.append((main_text, reference))
        if reference in paper_data["label"]:
            labels.append(1)
        else:
            labels.append(0)

from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

len(train_data), len(test_data)
