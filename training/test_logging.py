import torch

from collections import namedtuple
from torch.utils.data import DataLoader

from models import MultimodalSentimentModel, MultimodalTrainer

def test_logging():
  Batch = namedtuple('Batch', ['text_inputs', 'video_frames', 'audio_features'])
  
  mock_batch = Batch(
    text_inputs={'input_ids': torch.ones(1), 'attention_mask': torch.ones(1)},
    video_frames=torch.ones(1),
    audio_features=torch.ones(1)
  )
  
  mock_loader = DataLoader([mock_batch])
  
  model = MultimodalSentimentModel()
  trainer = MultimodalTrainer(model, mock_loader, mock_loader)
  
  train_losses = {
    'total': 0.5,
    'emotion': 0.3,
    'sentiment': 0.2
  }
  
  trainer.log_metrics(train_losses, phase="train")
  
  val_losses = {
    'total': 0.4,
    'emotion': 0.25,
    'sentiment': 0.15
  }
  
  val_metrics = {
    'emotion_precision': 0.8,
    'emotion_accuracy': 0.75,
    'sentiment_precision': 0.85,
    'sentiment_accuracy': 0.8
  }
  
  trainer.log_metrics(val_losses, val_metrics, phase="val")
  
  
if __name__ == "__main__":
  test_logging()