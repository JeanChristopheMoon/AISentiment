import pandas as pd
import json
import torch
import time
import os
from transformers import BartForSequenceClassification, BartTokenizer
from datetime import datetime

# Load model and tokenizer directly
print("Loading BART model for zero-shot classification...")
model_name = "facebook/bart-large-mnli"
model = BartForSequenceClassification.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Set device for computation
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)
print("Model loaded successfully")

# Define rhetorical strategy categories
rhetoric_categories = {
    "appeal_types": [
        "Expert Authority", "Institutional Authority", "Moral Authority", 
        "Experiential Authority", "Consensus Authority", "Appeal to Fear",
        "Appeal to Empathy", "Appeal to Pride", "Appeal to Guilt", "Appeal to Hope"
    ],
    "reasoning_types": [
        "Causal Reasoning", "Conditional Reasoning", "Analogical Reasoning",
        "Statistical Reasoning", "Historical Precedent"
    ],
    "fallacy_types": [
        "False Dichotomy", "Slippery Slope", "Ad Hominem", "Post Hoc Fallacy",
        "Straw Man", "Hasty Generalization"
    ],
    "framing_techniques": [
        "Metaphorical Framing", "Episodic Framing", "Thematic Framing",
        "Value Framing", "Risk Framing", "Reward Framing"
    ]
}

# Create appropriate hypothesis templates for each category
hypothesis_templates = {
    "appeal_types": "This text uses {label} as a persuasion technique.",
    "reasoning_types": "This text employs {label} in its argument.",
    "fallacy_types": "This text contains the {label} fallacy.",
    "framing_techniques": "This text uses {label} to present the issue."
}

# Also keep the original topic/tone/frame analysis
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

# Create a timestamp for output files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# This will store the results
results = []

# Process headlines using local model
print("Starting rhetorical strategy detection (using local model)...")
start_time = time.time()

# Process all headlines (or set a custom limit)
# headlines_to_process = headlines[:10]  # For testing with just 10
headlines_to_process = headlines  # Process all headlines

for i, headline in enumerate(headlines_to_process):
    print(f"🔍 Analyzing headline {i+1}/{len(headlines_to_process)}: {headline}")
    
    try:
        headline_analysis = {
            "headline": headline,
            "topic": {},
            "tone": {},
            "frame": {},
            "rhetoric": {}
        }
        
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
        headline_analysis["topic"] = {
            "top_match": topic_scores[0][0],
            "score": round(topic_scores[0][1], 4),
            "all_scores": [(label, round(score, 4)) for label, score in topic_scores]
        }
        
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
        headline_analysis["tone"] = {
            "top_match": tone_scores[0][0],
            "score": round(tone_scores[0][1], 4),
            "all_scores": [(label, round(score, 4)) for label, score in tone_scores]
        }
        
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
        headline_analysis["frame"] = {
            "top_match": frame_scores[0][0],
            "score": round(frame_scores[0][1], 4),
            "all_scores": [(label, round(score, 4)) for label, score in frame_scores]
        }
        
        # RHETORICAL STRATEGY ANALYSIS
        rhetorical_analysis = {}
        
        for category, labels in rhetoric_categories.items():
            category_scores = []
            template = hypothesis_templates[category]
            
            for label in labels:
                hypothesis = template.format(label=label)
                
                # Tokenize inputs
                inputs = tokenizer(headline, hypothesis, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                
                entailment_score = outputs.logits[0][2].item()
                category_scores.append((label, entailment_score))
            
            # Sort scores
            category_scores.sort(key=lambda x: x[1], reverse=True)
            rhetorical_analysis[category] = {
                "top_match": category_scores[0][0],
                "score": round(category_scores[0][1], 4),
                "all_scores": [(label, round(score, 4)) for label, score in category_scores]
            }
        
        headline_analysis["rhetoric"] = rhetorical_analysis
        
        # Store the complete analysis
        results.append(headline_analysis)
        
        # Print progress update with timing information
        elapsed = time.time() - start_time
        avg_time_per_headline = elapsed / (i + 1)
        remaining_headlines = len(headlines_to_process) - (i + 1)
        est_time_remaining = remaining_headlines * avg_time_per_headline
        
        print(f"  ✓ Analyzed headline {i+1}/{len(headlines_to_process)}")
        print(f"    Topic: {headline_analysis['topic']['top_match']} ({headline_analysis['topic']['score']:.4f})")
        print(f"    Tone: {headline_analysis['tone']['top_match']} ({headline_analysis['tone']['score']:.4f})")
        print(f"    Frame: {headline_analysis['frame']['top_match']} ({headline_analysis['frame']['score']:.4f})")
        print(f"    Rhetoric - Top appeal: {headline_analysis['rhetoric']['appeal_types']['top_match']}")
        print(f"    Rhetoric - Top fallacy: {headline_analysis['rhetoric']['fallacy_types']['top_match']}")
        print(f"    Time: {elapsed:.1f}s elapsed, ~{est_time_remaining/60:.1f} minutes remaining")
        
    except Exception as e:
        print(f"× Error processing headline: {e}")
    
    # Save results periodically to avoid losing progress
    if results and (i % 10 == 0 or i == len(headlines_to_process) - 1):
        progress_file = f"rhetorical_analysis_progress_{timestamp}.json"
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Progress saved: {len(results)}/{len(headlines_to_process)} headlines processed")

# Calculate total processing time
total_time = time.time() - start_time
print(f"Total processing time: {total_time/60:.2f} minutes")

# Save the final results
output_json = f"rhetorical_analysis_complete_{timestamp}.json"

print(f"Saving final results to {output_json}")
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Create a more structured CSV version with just the top matches
print("Creating CSV summary with top matches...")
csv_data = []

for item in results:
    row = {
        "headline": item["headline"],
        "topic": item["topic"]["top_match"],
        "topic_score": item["topic"]["score"],
        "tone": item["tone"]["top_match"],
        "tone_score": item["tone"]["score"],
        "frame": item["frame"]["top_match"],
        "frame_score": item["frame"]["score"]
    }
    
    # Add rhetorical strategies
    for category in rhetoric_categories:
        row[f"rhetoric_{category}"] = item["rhetoric"][category]["top_match"]
        row[f"rhetoric_{category}_score"] = item["rhetoric"][category]["score"]
    
    csv_data.append(row)

df_results = pd.DataFrame(csv_data)
output_csv = f"rhetorical_analysis_summary_{timestamp}.csv"
df_results.to_csv(output_csv, index=False)

print(f"✅ Process completed. {len(results)} headlines analyzed and saved.")
print(f"Full JSON results: {output_json}")
print(f"CSV summary: {output_csv}")

# Optional: Create a simple analysis of most common rhetorical strategies
print("\n--- Quick Summary of Results ---")
for category in rhetoric_categories:
    top_strategies = df_results[f"rhetoric_{category}"].value_counts().head(3)
    print(f"\nTop 3 {category}:")
    for strategy, count in top_strategies.items():
        percentage = (count / len(df_results)) * 100
        print(f"  {strategy}: {count} headlines ({percentage:.1f}%)")

print("\nAnalysis complete!")
