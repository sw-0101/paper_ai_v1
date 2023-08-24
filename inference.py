import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

with open("labeled_result_corrected.json", "r") as file:
    data = json.load(file)

model_path = "custom_bert_model.pth"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=100)

model.load_state_dict(torch.load(model_path))
model.eval()

def predict_importance(model, tokenizer, paper_title, abstract, references):
    concatenated_input = paper_title + " " + abstract + " " + " ".join(references)
    inputs = tokenizer(concatenated_input, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[0]
        scores = torch.sigmoid(logits).squeeze().tolist()

    return scores

with open("result.json", "r") as file:
    data = json.load(file)

paper_details = data.get("Attention Is All You Need", None)

if paper_details:
    abstract = paper_details["abstract"]
    references = paper_details["references"]
    scores = predict_importance(model, tokenizer, "Fractional-moment CAPM with loss aversion", abstract, references)
    sorted_references_scores = sorted(zip(references, scores), key=lambda x: x[1], reverse=True)

    for ref, score in sorted_references_scores:
        print(f"Reference: {ref}")
        print(f"Importance Score: {score:.4f}")
        print("----------")
else:
    print("The paper titled 'Attention Is All You Need' was not found in the dataset.")