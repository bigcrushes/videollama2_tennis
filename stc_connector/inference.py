import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from timm.models.regnet import RegStage
from timm.models.layers import LayerNorm, LayerNorm2d
import os
import einops
from transformers import (
    CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig,
)
import torch.nn.functional as F
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

hidden_size = 4096
mm_hidden_size = 768
num_hidden_layers= 32
NUM_FRAMES = 8
MAX_FRAMES = 64

cls = ['near_-_serve_-_last',
 'near_-_serve_B_in',
 'far_bh_return_DM_in',
 'near_bh_stroke_CC_in',
 'far_bh_stroke_-_last',
 'far_-_serve_T_in',
 'near_fh_return_-_last',
 'near_-_serve_W_in',
 'far_fh_return_DM_in',
 'near_fh_stroke_-_last',
 'near_-_serve_T_in',
 'far_bh_return_IO_in',
 'near_fh_stroke_DM_in',
 'far_fh_stroke_-_last',
 'far_-_serve_W_in',
 'near_fh_return_DM_in',
 'far_bh_stroke_CC_in',
 'near_fh_stroke_II_in',
 'far_fh_stroke_DM_in',
 'near_fh_stroke_IO_in',
 'far_bh_return_DL_in',
 'far_bh_return_CC_in',
 'near_fh_return_CC_in',
 'near_fh_return_IO_in',
 'far_bh_stroke_DM_in',
 'far_-_serve_B_in',
 'near_bh_return_DM_in',
 'near_fh_stroke_CC_in',
 'far_fh_stroke_CC_in',
 'near_bh_stroke_-_last',
 'far_bh_return_-_last',
 'near_bh_return_CC_in',
 'far_fh_stroke_II_in',
 'near_fh_stroke_DL_in',
 'far_fh_return_CC_in',
 'far_-_serve_-_last',
 'far_fh_return_-_last',
 'far_bh_return_II_in',
 'far_fh_stroke_IO_in',
 'far_fh_stroke_DL_in',
 'near_bh_stroke_DM_in',
 'far_bh_stroke_DL_in',
 'near_bh_return_-_last',
 'near_bh_stroke_DL_in',
 'near_bh_return_II_in',
 'near_bh_return_IO_in',
 'near_bh_stroke_IO_in',
 'near_fh_return_DL_in',
 'far_fh_return_DL_in',
 'near_bh_return_DL_in',
 'far_bh_stroke_IO_in',
 'near_fh_return_II_in',
 'far_fh_return_IO_in',
 'far_fh_return_II_in']

label_to_idx = {label: idx for idx, label in enumerate(set(cls))}

class Config:
    hidden_size = 4096
    mm_hidden_size = 1024
    num_hidden_layers= 32


class VideoDataset(Dataset):

    def __init__(self, labels, transform=None):
        self.labels = labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video, label = self.labels[idx]
        label = torch.tensor(label_to_idx[label]).to('cuda')
        images = process_video(video, image_processor)
        image_features = []
        for image in images:
            image_forward_out = vision_tower(image.to(device=self.device).unsqueeze(0), output_hidden_states=True)
            image_feature = feature_select(image_forward_out).to(image.dtype)
            image_features.append(image_feature)
            
        tensors = torch.stack(image_features, dim=0)
               
        return (tensors, label)

class STCConnector(nn.Module):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        """Temporal Convolutional Vision-Language Connector.
        
        Args:
            config: config object.
            downsample: (temporal, height, width) downsample rate.
            depth: depth of the spatial interaction blocks.
            mlp_depth: depth of the vision-language projector layers.
        """
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size
        # TODO: make these as config arguments
        self.depth = depth
        self.mlp_depth = mlp_depth
        self.downsample = downsample
        if depth != 0:
            self.s1 = RegStage(
                depth=depth,
                in_chs=encoder_hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s1 = nn.Identity()
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=1,
                bias=True
            ),
            nn.SiLU()
        )
        if depth != 0:
            self.s2 = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s2 = nn.Identity()
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)

        self.mlp = nn.Sequential(
              nn.Linear(5918720, 256),
              nn.Linear(256, 54)
        )

        
        
    def forward(self, x):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        t = x.size(1)
        if x.ndim == 4:
            hw = int(x.size(2) ** 0.5)
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        # 1. the first stage of the adapter
        x = self.s1(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler(x)
        new_t = x.size(2)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        x = self.readout(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        
        
        return x

def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def frame_sample(duration, mode='uniform', num_frames=None, fps=None):
    if mode == 'uniform':
        assert num_frames is not None, "Number of frames must be provided for uniform sampling."
        # NOTE: v1 version
        # Calculate the size of each segment from which a frame will be extracted
        seg_size = float(duration - 1) / num_frames

        frame_ids = []
        for i in range(num_frames):
            # Calculate the start and end indices of each segment
            start = seg_size * i
            end   = seg_size * (i + 1)
            # Append the middle index of the segment to the list
            frame_ids.append((start + end) / 2)

        return np.round(np.array(frame_ids) + 1e-6).astype(int)
        # NOTE: v0 version
        # return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == 'fps':
        assert fps is not None, "FPS must be provided for FPS sampling."
        segment_len = min(fps // NUM_FRAMES_PER_SECOND, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ImportError(f'Unsupported frame sampling mode: {mode}')


def process_video(video_path, processor, s=None, e=None, aspect_ratio='pad', num_frames=NUM_FRAMES):
    if isinstance(video_path, str):
        if s is not None and e is not None:
            s = s if s >= 0. else 0.
            e = e if e >= 0. else 0.
            if s > e:
                s, e = e, s
            elif s == e:
                e = s + 1

        # 1. Loading Video
        if os.path.isdir(video_path):
            frame_files = sorted(os.listdir(video_path))

            fps = 3
            num_frames_of_video = len(frame_files)
        elif video_path.endswith('.gif'):
            gif_reader = imageio.get_reader(video_path)

            fps = 25
            num_frames_of_video = len(gif_reader)
        else:
            vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)

            fps = vreader.get_avg_fps()
            num_frames_of_video = len(vreader)

        # 2. Determine frame range & Calculate frame indices
        f_start = 0                       if s is None else max(int(s * fps) - 1, 0)
        f_end   = num_frames_of_video - 1 if e is None else min(int(e * fps) - 1, num_frames_of_video - 1)
        frame_indices = list(range(f_start, f_end + 1))

        duration = len(frame_indices)
        # 3. Sampling frame indices
        if num_frames is None:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='fps', fps=fps)]
        else:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=num_frames)]

        # 4. Acquire frame data
        if os.path.isdir(video_path):
            video_data = [Image.open(os.path.join(video_path, frame_files[f_idx])) for f_idx in sampled_frame_indices]
        elif video_path.endswith('.gif'):
            video_data = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)) for idx, frame in enumerate(gif_reader) if idx in sampled_frame_indices]
        else:
            video_data = [Image.fromarray(frame) for frame in vreader.get_batch(sampled_frame_indices).asnumpy()]

    elif isinstance(video_path, np.ndarray):
        video_data = [Image.fromarray(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], np.ndarray):
        video_data = [Image.fromarray(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], str):
        video_data = [Image.open(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], Image.Image):
        video_data = video_path
    else:
        raise ValueError(f"Unsupported video path type: {type(video_path)}")

    while num_frames is not None and len(video_data) < num_frames:
        video_data.append(Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8)))

    # MAX_FRAMES filter
    video_data = video_data[:MAX_FRAMES]

    if aspect_ratio == 'pad':
        images = [expand2square(f, tuple(int(x*255) for x in processor.image_mean)) for f in video_data]
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']
    else:
        images = [f for f in video_data]
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']
    return video

select_layer = -1
def feature_select(image_forward_outs):
    image_features = image_forward_outs.hidden_states[select_layer]
    image_features = image_features[:, 1:]

    return image_features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
vision_tower.to(device)
vision_tower.requires_grad_(False)

labels = []
cls = []
df = pd.read_json("test.json")
for idx, row in df.iterrows():
    for event in row['events']:
        path = f"../new/VideoLLaMA2/test_datasets/{event['label']}/{row['video']}_{str(event['frame'])}.mp4"
        if os.path.exists(path):
            labels.append((path, event['label']))

vds = VideoDataset(labels)
test_dataloader = DataLoader(vds, batch_size=4, shuffle=True)

print("Dataset loaded", flush=True)

model = STCConnector(Config()).to(device)
model.load_state_dict(torch.load('./final_model_stc2', weights_only=True))
model.eval()
criterion = nn.CrossEntropyLoss()

y_pred = []
y_true = []

with torch.no_grad():
    for input_data in test_dataloader:
        x, y = input_data
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        pred = torch.argmax(output, dim=-1)
        for i in range(len(pred.cpu())):
            print(cls[pred[i].cpu().item()], cls[y[i].cpu().item()], flush=True)
        print(criterion(output, y), flush=True)
        y_pred.append(pred.cpu().numpy())
        y_true.append(y.cpu().numpy())

y_pred = np.concatenate(y_pred, axis=0)
y_true = np.concatenate(y_true, axis=0)

print(y_pred)
print('--------------------------------------')
print(y_true)

print(accuracy_score(y_true, y_pred))
