from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_intent(text, labels=None):
    if labels is None:
        labels = ["billing issue", "technical issue", "product inquiry", "general feedback"]
    result = classifier(text, labels)
    top_label = result["labels"][0]
    confidence = result["scores"][0]
    return top_label, confidence

if __name__ == "__main__":
    sample = "My app keeps crashing every time I open the camera"
    label, score = classify_intent(sample)
    print(f"Intent: {label} (confidence: {score:.2f})")