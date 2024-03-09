import os
import json
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')



class TransformerDataset(Dataset):
  def __init__(self, csv_dir, image_dir, image_encoder):
    self.csv_dir = csv_dir
    self.image_dir = image_dir
    self.df = pd.read_csv(csv_dir)
    
    self.transform = transforms.Compose([transforms.Resize(232) if image_encoder == 'resnet50' else transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    headline = self.df.iloc[index]['clean_title']
    image_dir = os.path.join(self.image_dir, self.df.iloc[index]['id'] + '.jpg')
    class_label = self.df.iloc[index]['2_way_label']
    domain_label = self.df.iloc[index]['subreddit_label']
    
    try:
      image = Image.open(image_dir).convert('RGB')
      processed_image = self.transform(image).view(1, 3, 224, 224)
    except:
      return None
    
    return {'news': (headline, processed_image), 'label': (class_label, domain_label)}
    

def custom_collate_fn(batch):
  headlines = []
  images = []
  class_labels = []
  domain_labels = []

  for item in batch:
    if item != None:
      headlines.append(item['news'][0])
      images.append(item['news'][1])
      class_labels.append(item['label'][0])
      domain_labels.append(item['label'][1])

  class_labels = torch.tensor(class_labels, dtype=torch.int64)
  domain_labels = torch.tensor(domain_labels, dtype=torch.int64)

  return (headlines, images), (class_labels, domain_labels)