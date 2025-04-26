import pandas as pd
import json
import torch
import time
import os
from transformers import BartForSequenceClassification, BartTokenizer

# Load model and tokenizer directly
model_name = "facebook/bart-large-mnli"
model = BartForSequenceClassification.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Set device for computation
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)
print("Model loaded successfully")

# Define candidate labels for classification
topic_labels = ["Economy", "Foreign Policy", "Human Rights", "Environment", "Security", "Technology", "EU Governance"]
tone_labels = ["Neutral", "Urgent", "Optimistic", "Conflict-Oriented", "Critical", "Supportive"]
frame_labels = ["Humanitarian", "Security", "Legalistic", "Economic", "Nationalist", "Technocratic"]

# Check if the file exists before starting
file_path = "europarl_headlines_max_5000.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Data file '{file_path}' not found. Please ensure the file exists in the current directory.")

# Read your headlines
print(f"Reading data from {file_path}")
df = pd.read_csv(file_path)
headlines = df["Headline"].tolist()
print(f"Loaded {len(headlines)} headlines from CSV file")

# This will store the results
results = []

# Process headlines using local model
print("Starting classification process (using local model)...")

# Define how many headlines to process (e.g., first 100)
headlines_to_process = headlines[:100]

for i, headline in enumerate(headlines_to_process):
    print(f"🔍 Classifying headline {i+1}/{len(headlines_to_process)}: {headline}")
    
    try:
        # TOPIC CLASSIFICATION
        topic_scores = []
        for label in topic_labels:
            # Create hypothesis for zero-shot classification
            hypothesis = f"This text is about {label}."
            
            # Tokenize inputs
            inputs = tokenizer(headline, hypothesis, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
            
            # BART gives entailment vs contradiction scores (2 = entailment)
            entailment_score = outputs.logits[0][2].item()
            topic_scores.append((label, entailment_score))
        
        # Sort by score in descending order
        topic_scores.sort(key=lambda x: x[1], reverse=True)
        top_topic = topic_scores[0][0]
        top_topic_score = topic_scores[0][1]
        
        # TONE CLASSIFICATION
        tone_scores = []
        for label in tone_labels:
            hypothesis = f"The tone of this text is {label}."
            inputs = tokenizer(headline, hypothesis, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            entailment_score = outputs.logits[0][2].item()
            tone_scores.append((label, entailment_score))
        
        tone_scores.sort(key=lambda x: x[1], reverse=True)
        top_tone = tone_scores[0][0]
        top_tone_score = tone_scores[0][1]
        
        # FRAME CLASSIFICATION
        frame_scores = []
        for label in frame_labels:
            hypothesis = f"This text uses a {label} frame."
            inputs = tokenizer(headline, hypothesis, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            entailment_score = outputs.logits[0][2].item()
            frame_scores.append((label, entailment_score))
        
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        top_frame = frame_scores[0][0]
        top_frame_score = frame_scores[0][1]
        
        # Store the result
        results.append({
            "headline": headline,
            "topic": top_topic,
            "topic_confidence": round(top_topic_score, 4),
            "tone": top_tone, 
            "tone_confidence": round(top_tone_score, 4),
            "frame": top_frame,
            "frame_confidence": round(top_frame_score, 4)
        })
        
        print(f"  ✓ Classified as Topic: {top_topic} ({top_topic_score:.4f}), " 
              f"Tone: {top_tone} ({top_tone_score:.4f}), "
              f"Frame: {top_frame} ({top_frame_score:.4f})")
        
    except Exception as e:
        print(f"× Error processing headline: {e}")
        
    # Save results periodically to avoid losing progress
    if results and (i % 10 == 0 or i == len(headlines_to_process) - 1):
        with open("labeled_headlines_progress.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Progress saved: {len(results)} headlines processed so far")

# Save the final results
output_json = "labeled_headlines.json"
output_csv = "structured_labeled_headlines.csv"

print(f"Saving results to {output_json} and {output_csv}")

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Save as CSV for further analysis
if results:
    df_labels = pd.DataFrame(results)
    df_labels.to_csv(output_csv, index=False)
    print(f"✅ Process completed. {len(results)} headlines classified and saved locally.")
else:
    print("⚠️ No results were collected. Check the errors above.")

print("Classification process complete!")
