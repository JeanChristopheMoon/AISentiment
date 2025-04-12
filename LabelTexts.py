import requests
import pandas as pd
import json
import time
import os

# Set up your Hugging Face API Key and model
API_TOKEN = "YourApiCode"
MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

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
    "Authorization": f"Bearer {API_TOKEN}"
}

# This will store the results
results = []

# Process headlines - sending each one to Hugging Face servers
print("Starting classification process (runs on Hugging Face servers)...")
for i, headline in enumerate(headlines[:100]):  # Start with first 100 headlines
    print(f"🔍 Classifying headline {i+1}/{min(100, len(headlines))}: {headline}")
    
    try:
        # Prepare the request payload for Hugging Face API (zero-shot classification)
        payload = {
            "inputs": headline,
            "parameters": {
                "candidate_labels": topic_labels
            }
        }

        # Send POST request to Hugging Face Inference API
        response_topic = requests.post(MODEL_URL, headers=headers, json=payload)
        topic_result = response_topic.json()

        # Debugging: Print the full response to check the structure
        print("Topic Result:", topic_result)

        # Ensure that 'labels' and 'scores' keys exist in the response
        if 'labels' in topic_result[0] and 'scores' in topic_result[0]:
            top_topic = topic_result[0]['labels'][0]
            top_topic_score = topic_result[0]['scores'][0]

            # Repeat for tone
            payload["parameters"]["candidate_labels"] = tone_labels
            response_tone = requests.post(MODEL_URL, headers=headers, json=payload)
            tone_result = response_tone.json()

            # Debugging: Print the full response to check the structure
            print("Tone Result:", tone_result)

            if 'labels' in tone_result[0] and 'scores' in tone_result[0]:
                top_tone = tone_result[0]['labels'][0]
                top_tone_score = tone_result[0]['scores'][0]

                # Repeat for frame
                payload["parameters"]["candidate_labels"] = frame_labels
                response_frame = requests.post(MODEL_URL, headers=headers, json=payload)
                frame_result = response_frame.json()

                # Debugging: Print the full response to check the structure
                print("Frame Result:", frame_result)

                if 'labels' in frame_result[0] and 'scores' in frame_result[0]:
                    top_frame = frame_result[0]['labels'][0]
                    top_frame_score = frame_result[0]['scores'][0]

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

                    print(f"  ✓ Classified as Topic: {top_topic}, Tone: {top_tone}, Frame: {top_frame}")
                else:
                    print("Error in frame classification response:", frame_result)
            else:
                print("Error in tone classification response:", tone_result)
        else:
            print("Error in topic classification response:", topic_result)

    except Exception as e:
        print(f"Error processing headline: {e}")

    time.sleep(1)  # Sleep to avoid rate-limiting on the API

# Save the results locally
output_json = "labeled_headlines.json"
output_csv = "structured_labeled_headlines.csv"

print(f"Saving results locally to {output_json} and {output_csv}")

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Optionally: Save the data into a CSV for further analysis
df_labels = pd.DataFrame(results)
df_labels.to_csv(output_csv, index=False)

print(f"✅ Process completed. {len(results)} headlines classified and saved locally.")
print(f"Summary: Data was read locally, processed on Hugging Face servers, and results saved locally.")
