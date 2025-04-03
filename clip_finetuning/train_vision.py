## Frame averaging

import os

print("In script", flush=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["WORLD_SIZE"] = "1"

print("OS set", flush=True)

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
print("Loaded torch", flush=True)
from transformers import CLIPImageProcessor, CLIPVisionModel
print("Loaded CLIP", flush=True)
from PIL import Image
import torch.nn.functional as F
import numpy as np

print("Loaded all modules", flush=True)

# Step 1: Extract frames from videos and resize them
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
                frames.append(frames[-1])  # Repeat last frame if video ends
            else:
                frames.append(torch.zeros((target_size[0], target_size[1], 3)))  # If no frame, append empty frame

    # If the video is too short, pad with the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])

    cap.release()
    return frames

# Step 2: Custom Dataset for Video
class VideoDataset(Dataset):
    def __init__(self, csv_file, processor, num_frames=16, target_size=(336, 336)):
        self.data = pd.read_csv(csv_file)
        self.processor = processor
        self.num_frames = num_frames
        self.target_size = target_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data.iloc[idx, 0]  # Path to the video
        text = self.data.iloc[idx, 1]        # Corresponding text label

        # Extract and pad frames to ensure uniformity
        frames = extract_frames(video_path, self.num_frames, target_size=self.target_size)
        
        # Average the frames along the time dimension to create a single image
        averaged_frame = np.mean(frames, axis=0).astype(np.uint8)

        # Process text and the averaged image
        inputs = self.processor(
            text=text, 
            images=[Image.fromarray(averaged_frame)], 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),   # Text input IDs
            'attention_mask': inputs['attention_mask'].squeeze(0), 
            'pixel_values': inputs['pixel_values'].squeeze(0),  # Processed frame values (now 4D)
            'label': idx                                   # Using index as label for now (replace as needed)
        }

# Step 3: Custom collate_fn for DataLoader
def custom_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]

    # Pad input_ids and attention_mask to ensure uniform sequence length
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Stack pixel_values (image frames)
    pixel_values = torch.stack(pixel_values)

    labels = torch.tensor([item['label'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'label': labels
    }

# Step 4: Load Model and Processor
print("Loading model...", flush=True)
model = CLIPVisionModel.from_pretrained("./clip-vit-large-patch14-336")
processor = CLIPImageProcessor.from_pretrained("./clip-vit-large-patch14-336")
print("Model loaded", flush=True)
for name, param in model.named_parameters():
    print(name, param.requires_grad)

# Step 5: Create Dataset and DataLoader
dataset = VideoDataset('out2.csv', processor, num_frames=64, target_size=(336, 336))
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
print("Data loaded", flush=True)

# Step 6: Training Loop
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

num_epochs = 1  # Adjust as needed

model.train()
for epoch in range(num_epochs):
    for idx, batch in enumerate(dataloader):
        print(f"Batch {idx}/{len(dataloader)}", flush=True)
        images = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass
        outputs = model(pixel_values=images)
        logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text

        # Contrastive loss
        labels = torch.arange(len(images)).to(device)
        loss_img = torch.nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_text = torch.nn.CrossEntropyLoss()(logits_per_text, labels)
        loss = (loss_img + loss_text) / 2

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}",flush=True)

# Step 7: Save the Model
model.save_pretrained('./fine_tuned_clip_vision')
processor.save_pretrained('./fine_tuned_clip_processor_vision')

print("Model saved to fine_tuned_clip_vision!")
