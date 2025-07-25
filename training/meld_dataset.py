import subprocess
import cv2
import numpy as np
import pandas as pd
import torch
import torchaudio
import os

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class MELDDataset(Dataset):
  def __init__(self, csv_path, video_dir):
    self.data = pd.read_csv(csv_path)
    self.video_dir = video_dir

    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # neutral, joy, sadness, anger, surprise, fear, disgust
    self.emotion_map = {
      'neutral': 0,
      'joy': 1,
      'sadness': 2,
      'anger': 3,
      'surprise': 4,
      'fear': 5,
      'disgust': 6
    }

    # positive, neutral, negative
    self.sentiment_map = {
      'positive': 0,
      'neutral': 1,
      'negative': 2
    }

  def _load_video_frames(self, video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
      if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
      
      # Try to read first frame to validate video
      ret, frame = cap.read()
      
      if not ret or frame is None:
        raise ValueError(f"Failed to read frames from video: {video_path}")
      
      # Reset index after reading the first frame
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      
      # Fetch only 30 frames => guarantee video is consistent
      while len(frames) < 30 and cap.isOpened():
        ret, frame = cap.read()
        if not ret: break # if no more frames
        
        frame = cv2.resize(frame, (224, 224))  # Resize to 224x224
        frame = frame / 255.0  # Normalize the frame to [0, 1] (e.g [255, 124, 0] => [1.0, 0.486, 0])
        frames.append(frame)
        
    except Exception as e:
      raise ValueError(f"Error loading video frames: {str(e)}")
    finally:
      cap.release()
      
    if (len(frames) == 0):
      raise ValueError(f"No frames extracted from video: {video_path}")
    
    # Pad or truncarte to 30 frames
    if len(frames) < 30:
      frames += [np.zeros_like(frames[0])] * (30 - len(frames))  # Pad with zeros
    else:
      frames = frames[:30] # Truncate to 30 frames
    
    # Before permute: [frames, height, width, channels] (0, 1, 2, 3)
    # After permute:  [frames, channels, height, width] (0, 3, 1, 2)
    return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

  def _extract_aduio_features(self, video_path):
    audio_path = video_path.replace('.mp4', '.wav')
    
    try:
      subprocess.run([
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # Audio codec
        '-ar', '16000',  # Sample rate
        '-ac', '1',  # Mono channel
        audio_path
      ], check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
      
      wavefrom, sample_rate = torchaudio.load(audio_path)
      
      if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        wavefrom = resampler(wavefrom)
      
      mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=64,
        n_fft=1024,
        hop_length=512,
      )
      
      mel_spec = mel_spectrogram(wavefrom)
      
      # Normalize the mel spectrogram
      mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
      
      if mel_spec.size(2) < 300:
        padding = 300 - mel_spec.size(2)
        mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
      else:
        mel_spec = mel_spec[:, :, :300] # Keep channels, frequency bins, and only 300 timestamps
      
      return mel_spec
        
    except subprocess.CalledProcessError as e:
      raise ValueError(f"FFmpeg error: {str(e)}") 
      
    except Exception as e:
      raise ValueError(f"Error extracting audio features: {str(e)}")
    
    finally:
      if os.path.exists(audio_path):
        os.remove(audio_path)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if isinstance(idx, torch.Tensor):
      idx = idx.item()
    
    row = self.data.iloc[idx]
    
    try:
      video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""  # fetch video filename

      path = os.path.join(self.video_dir, video_filename)
      video_path = os.path.exists(path)

      if video_path == False:
        raise FileNotFoundError(f"Video file {video_filename} not found in {self.video_dir}")

      print(f"Processing video: {video_filename}")

      text_inputs = self.tokenizer(
        row['Utterance'],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
      )
      
      video_frames = self._load_video_frames(path)
      audio_features = self._extract_aduio_features(path)
      
      # Map sentiment and emotion labels
      emotion_label = self.emotion_map[row['Emotion'].lower()]
      sentiment_label = self.sentiment_map[row['Sentiment'].lower()]
      
      print(f"Emotion label: {emotion_label}, Sentiment label: {sentiment_label}")
      
      return {
        'text_inputs': {
          'input_ids': text_inputs['input_ids'].squeeze(0),
          'attention_mask': text_inputs['attention_mask'].squeeze(0)
        },
        'video_frames': video_frames,
        'audio_features': audio_features,
        'emotion_label': torch.tensor(emotion_label),
        'sentiment_label': torch.tensor(sentiment_label)
      }
      
    except Exception as e:
      raise ValueError(f"Error processing {path}: {str(e)}")
      return None

def collate_fn(batch):
  # Filter out None samples
  batch = list(filter(None, batch))
  return torch.utils.data.dataloader.default_collate(batch)

def preprare_data_loaders(train_csv, train_video_dir,
                          dev_csv, dev_video_dir,
                          test_csv, test_video_dir, batch_size=32):
  train_dataset = MELDDataset(train_csv, train_video_dir)
  dev_dataset = MELDDataset(test_csv, test_video_dir)
  test_dataset = MELDDataset(dev_csv, dev_video_dir)
  
  train_loader = DataLoader(train_dataset,
                            batch_size=batch_size, 
                            shuffle=True,
                            collate_fn=collate_fn)
  
  dev_loader = DataLoader(dev_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          collate_fn=collate_fn)
  
  test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn)
  
  return train_loader, dev_loader, test_loader
  
# Onload the dataset
if __name__ == "__main__":
  train_loader, dev_loader, test_loader = preprare_data_loaders(
    "data/train/train_sent_emo.csv", "data/train/train_splits",
    "data/dev/dev_sent_emo.csv", "data/dev/dev_splits_complete",
    "data/test/test_sent_emo.csv", "data/test/output_repeated_splits_test"
  )

  for batch in train_loader:
    print(batch['text_inputs'])
    print(batch['video_frames'].shape)
    print(batch['video_frames'].shape)
    print(batch['emotion_label'])
    print(batch['sentiment_label'])
    break  # Just to test the first batch