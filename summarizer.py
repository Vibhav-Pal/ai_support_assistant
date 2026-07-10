from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=60, min_length=20):
    if len(text.split()) < 30:
        return text  # too short to summarize meaningfully
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["summary_text"]

if __name__ == "__main__":
    sample = """
    Customers have been reporting that the mobile app freezes when switching
    between the video call screen and the chat screen. This started after
    the last update was pushed to production. Several users on Android 13
    devices are affected, while iOS users report no issues. The engineering
    team is currently investigating the root cause and expects a fix within
    the next release cycle.
    """
    print(summarize_text(sample))