import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["WORLD_SIZE"] = "1"

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Step 1: Extract frames from video (same as during training)
def extract_frames(video_path, num_frames=16, target_size=(336, 336)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // num_frames)  # Step size to select frames

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, target_size)
            frames.append(resized_frame)
        else:
            if len(frames) > 0:
                frames.append(frames[-1])  # Repeat last frame if video ends early
            else:
                frames.append(np.zeros((target_size[0], target_size[1], 3)))  # Append an empty frame if no frames

    # If the video is too short, pad with the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])

    cap.release()
    return frames

# Step 2: Load the fine-tuned model and processor
model = CLIPModel.from_pretrained('./fine_tuned_clip')  # Path to your saved fine-tuned model
processor = CLIPProcessor.from_pretrained('./fine_tuned_clip_processor')  # Path to your processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Step 3: Inference function
def predict_video(video_path, text_labels, num_frames=64, target_size=(336, 336)):
    # Extract frames from video
    frames = extract_frames(video_path, num_frames=num_frames, target_size=target_size)

    averaged_frame = np.mean(frames, axis=0).astype(np.uint8)

    # Process frames and text
    inputs = processor(
        text=text_labels, 
        images=[Image.fromarray(averaged_frame)], 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )

    # Move inputs to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    pixel_values = inputs['pixel_values'].to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        logits_per_image = outputs.logits_per_image  # Image-to-text logits
        
    # Aggregate logits across frames
    batch_size, num_frames = logits_per_image.shape
    #logits_per_image = logits_per_image.view(batch_size, num_frames, -1).mean(dim=1)  # Mean pool over frames
    
    # Get predicted text for each video
    predicted_idx = logits_per_image.argmax(dim=-1).item()  # Index of highest score
    predicted_label = text_labels[predicted_idx]  # Retrieve corresponding text label

    return predicted_label

# Step 4: Test the inference with a sample video and text labels
if __name__ == "__main__":
    video_path = "../new/VideoLLaMA2/test_datasets/"
 
    text_labels = ['The near player hit a serve last.',
       'The near player hit a wide serve in.',
       'The far player hit a backhand cross-court return in.',
       'The near player hit a forehand cross-court stroke in.',
       'The far player hit a backhand down the line stroke in.',
       'The near player hit a backhand stroke last.',
       'The far player hit a backhand down the middle return in.',
       'The far player hit a backhand cross-court stroke in.',
       'The far player hit a backhand down the middle stroke in.',
       'The near player hit a backhand down the middle stroke in.',
       'The far player hit a backhand stroke last.',
       'The far player hit a forehand return last.',
       'The far player hit a wide serve in.',
       'The near player hit a forehand down the middle return in.',
       'The far player hit a forehand cross-court stroke in.',
       'The far player hit a forehand down the middle stroke in.',
       'The near player hit a forehand stroke last.',
       'The far player hit a T serve in.',
       'The near player hit a forehand return last.',
       'The far player hit a serve last.',
       'The near player hit a backhand return last.',
       'The far player hit a body serve in.',
       'The near player hit a T serve in.',
       'The near player hit a forehand down the middle stroke in.',
       'The far player hit a forehand stroke last.',
       'The far player hit a forehand cross-court return in.',
       'The near player hit a backhand cross-court stroke in.',
       'The far player hit a backhand return last.',
       'The near player hit a backhand down the middle return in.',
       'The far player hit a forehand inside out stroke in.',
       'The near player hit a forehand inside out stroke in.',
       'The far player hit a backhand inside out return in.',
       'The far player hit a backhand down the line return in.',
       'The far player hit a forehand inside out return in.',
       'The near player hit a forehand down the line stroke in.',
       'The near player hit a backhand inside out stroke in.',
       'The near player hit a forehand cross-court return in.',
       'The near player hit a body serve in.',
       'The near player hit a backhand down the line return in.',
       'The far player hit a backhand inside out stroke in.',
       'The near player hit a backhand cross-court return in.',
       'The near player hit a backhand down the line stroke in.',
       'The far player hit a forehand inside in stroke in.',
       'The near player hit a forehand inside in stroke in.',
       'The far player hit a backhand inside in return in.',
       'The near player hit a backhand inside out return in.',
       'The far player hit a forehand down the middle return in.',
       'The near player hit a forehand down the line return in.',
       'The near player hit a forehand inside out return in.',
       'The far player hit a forehand down the line stroke in.',
       'The near player hit a backhand inside in return in.',
       'The far player hit a forehand down the line return in.',
       'The far player hit a forehand inside in return in.',
       'The near player hit a forehand inside in return in.']

    
    preds = {}

    for label in os.listdir(video_path):
        temp = os.path.join(video_path, label)
        if os.path.isfile(temp):
            continue
        for video in os.listdir(temp):
            temp2 = os.path.join(temp, video)
            print(temp2, flush=True) 
            predicted_label = predict_video(temp2, text_labels)
            print(f"Predicted label: {predicted_label}\nTrue label: {label}", flush=True)

            preds[video] = (predicted_label, label)

    print(preds)

