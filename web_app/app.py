from flask import Flask, render_template, request
import pandas as pd
from PIL import Image
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'models'))
sys.path.append(os.path.join(os.getcwd(), 'utils'))
from news_net3 import NewsNet3
import torch
import load_configs
from transformers import BertTokenizer as TextTokenizer
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")



# Create a Flask application
app = Flask(__name__)
app.debug = True

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available on this system.")
    else:
        device = 'cpu'
        print("CUDA is not available on this system.")

_, model_config, _ = load_configs.load()

dtype = torch.float32
headline_len = 50
model_params = {
    'text_encoder': model_config['text_encoder'],
    'text_encoder_trainable': model_config['text_encoder_trainable'],
    'n_trainable_layers': model_config['n_trainable_layers'],
    'image_encoder': model_config['image_encoder'][1],
    'image_encoder_trainable': model_config['image_encoder_trainable'],
    'dropout_prob': model_config['dropout_prob'],
    'repr_size':  model_config['repr_size'],
    'lambd': model_config['lambd']
}

tokenizer = TextTokenizer.from_pretrained(model_params['text_encoder'])
model_state_dict_dir = os.path.join(os.getcwd(), 'results', 'resnet50', 'best_model_6.pth')
model = NewsNet3(model_params).to(dtype).to(device)
model.load_state_dict(torch.load(model_state_dict_dir))

labels = {0: 'true', 1: 'fake'}

# Define a route for home page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Define a route for csv upload
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['input_file']
    if str(file).find('.tsv'):
        df = pd.read_csv(file, sep='\t')
    else:
        df = pd.read_csv(file)
    items = []
    sample_size = len(df)
    accuracy = 0
    for _, row in df.iterrows():
        # Process text
        tokenized_headline = tokenizer.encode_plus(
                row['clean_title'],
                add_special_tokens=True,
                max_length=headline_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
        ).to(device)
        
        # Process image
        image_dir = os.path.join(os.getcwd(), 'datasets', 'images/') + row['id'] + '.jpg'
        try:
            image = Image.open(image_dir).convert('RGB')  
            transform = transforms.Compose([transforms.Resize(232),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            input_image = transform(image).view(1, 3, 224, 224).to(dtype).to(device)
        
            input_X = (tokenized_headline, input_image)
            input_y = torch.tensor(row['2_way_label'])
            input_y = torch.squeeze(input_y).to(dtype).to(device)
            
            # Apply the model
            model.eval()
            pred_y = model(input_X)
            
            pred_y = torch.squeeze(pred_y)
            prediction = torch.eq((pred_y > 0.5), input_y).to(dtype)
            accuracy += torch.mean(prediction, dtype=dtype).item()
            pred_y = (pred_y > 0.5).to(dtype)
        
            resized_image = image.resize((256, 256))
            image_dir = os.path.join(os.getcwd(), 'web_app', 'static/') + row['id'] + '.jpg'
            resized_image.save(image_dir)
            item = {
                'image': row['id'] + '.jpg',
                'title': row['clean_title'],
                'true label': labels[row['2_way_label']],
                'predicted label': labels[int(pred_y.item())]
            }
            items.append(item)
        except:
            continue
    
    return render_template('index.html', items=items, sample_size=sample_size, accuracy=round(accuracy / sample_size, 3))

if __name__ == '__main__':
    app.run()