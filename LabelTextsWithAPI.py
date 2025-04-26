import requests
import pandas as pd
import json
import time
import os

# Set up your Hugging Face API Key and model
API_TOKEN = "YourAPI"
# Using a better model specifically for zero-shot classification
MODEL_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

# Check if the file exists before starting
file_path = "europarl_headlines_max_5000.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Data file '{file_path}' not found. Please ensure the file exists in the current directory.")

# Define candidate labels for classification
topic_labels = ["Economy", "Foreign Policy", "Human Rights", "Environment", "Security", "Technology", "EU Governance"]
tone_labels = ["Neutral", "Urgent", "Optimistic", "Conflict-Oriented", "Critical", "Supportive"]
frame_labels = ["Humanitarian", "Security", "Legalistic", "Economic", "Nationalist", "Technocratic"]

# Read your headlines locally
print(f"Reading data from {file_path}")
df = pd.read_csv(file_path)
headlines = df["Headline"].tolist()
print(f"Loaded {len(headlines)} headlines from CSV file")

# Prepare headers for API request
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# This will store the results
results = []

# Process headlines - sending each one to Hugging Face servers
print("Starting classification process (runs on Hugging Face servers)...")
for i, headline in enumerate(headlines[:100]):  # Start with first 100 headlines
    print(f"🔍 Classifying headline {i+1}/{min(100, len(headlines))}: {headline}")
    
    try:
        # Prepare the request payload for BART zero-shot classification
        payload = {
            "inputs": headline,
            "parameters": {
                "candidate_labels": topic_labels
            }
        }
        
        # Send POST request for topic classification
        topic_response = requests.post(MODEL_URL, headers=headers, json=payload)
        
        # Check for successful response
        if topic_response.status_code == 200:
            topic_result = topic_response.json()
            
            # Extract topic classification
            if "labels" in topic_result and "scores" in topic_result:
                top_topic = topic_result["labels"][0]
                top_topic_score = topic_result["scores"][0]
                
                # Continue with tone classification
                payload["parameters"]["candidate_labels"] = tone_labels
                tone_response = requests.post(MODEL_URL, headers=headers, json=payload)
                
                if tone_response.status_code == 200:
                    tone_result = tone_response.json()
                    
                    if "labels" in tone_result and "scores" in tone_result:
                        top_tone = tone_result["labels"][0]
                        top_tone_score = tone_result["scores"][0]
                        
                        # Continue with frame classification
                        payload["parameters"]["candidate_labels"] = frame_labels
                        frame_response = requests.post(MODEL_URL, headers=headers, json=payload)
                        
                        if frame_response.status_code == 200:
                            frame_result = frame_response.json()
                            
                            if "labels" in frame_result and "scores" in frame_result:
                                top_frame = frame_result["labels"][0]
                                top_frame_score = frame_result["scores"][0]
                                
                                # Store the result locally
                                results.append({
                                    "headline": headline,
                                    "topic": top_topic,
                                    "topic_confidence": round(top_topic_score, 4),
                                    "tone": top_tone,
                                    "tone_confidence": round(top_tone_score, 4),
                                    "frame": top_frame,
                                    "frame_confidence": round(top_frame_score, 4)
                                })
                                
                                print(f"  ✓ Classified as Topic: {top_topic} ({top_topic_score:.4f}), Tone: {top_tone} ({top_tone_score:.4f}), Frame: {top_frame} ({top_frame_score:.4f})")
                            else:
                                print(f"  × Error: Missing expected keys in frame response: {frame_result}")
                        else:
                            print(f"  × Error in frame API call: {frame_response.status_code}")
                            if frame_response.status_code == 429:
                                print("  Rate limit exceeded. Waiting 60 seconds before continuing...")
                                time.sleep(60)
                                continue  # Retry this headline
                    else:
                        print(f"  × Error: Missing expected keys in tone response: {tone_result}")
                else:
                    print(f"  × Error in tone API call: {tone_response.status_code}")
                    if tone_response.status_code == 429:
                        print("  Rate limit exceeded. Waiting 60 seconds before continuing...")
                        time.sleep(60)
                        continue  # Retry this headline
            else:
                print(f"  × Error: Missing expected keys in topic response: {topic_result}")
        else:
            print(f"  × Error in topic API call: {topic_response.status_code}")
            if topic_response.status_code == 429:
                print("  Rate limit exceeded. Waiting 60 seconds before continuing...")
                time.sleep(60)
                continue  # Retry this headline
            elif topic_response.status_code == 503:
                print("  Model still loading. Waiting 10 seconds...")
                time.sleep(10)
                continue  # Retry this headline

    except Exception as e:
        print(f"× Error processing headline: {e}")

    # Save results after each successful classification to avoid losing progress
    if results and (i % 10 == 0 or i == len(headlines[:100]) - 1):  # Save every 10 processed headlines and at the end
        with open("labeled_headlines_progress.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Progress saved: {len(results)} headlines processed so far")

    time.sleep(1)  # Wait between requests to avoid rate limiting, adjust as needed

# Save the final results locally
output_json = "labeled_headlines.json"
output_csv = "structured_labeled_headlines.csv"

print(f"Saving results locally to {output_json} and {output_csv}")

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Optionally: Save the data into a CSV for further analysis
if results:  # Only create DataFrame if we have results
    df_labels = pd.DataFrame(results)
    df_labels.to_csv(output_csv, index=False)
    print(f"✅ Process completed. {len(results)} headlines classified and saved locally.")
else:
    print("⚠️ No results were collected. Check the errors above.")

print(f"Summary: Data was read locally, processed on Hugging Face servers, and results saved locally.")
