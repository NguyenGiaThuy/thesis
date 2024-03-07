import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from transformers import BertModel



class BertLayer(nn.Module):
    def __init__(self, n_fine_tune_layers=10):
        super().__init__()
        self.n_fine_tune_layers = n_fine_tune_layers
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        for name, param in self.bert.named_parameters():
            if 'encoder.layer.' in name and int(name.split('.')[2]) < self.bert.config.num_hidden_layers - self.n_fine_tune_layers:
                param.requires_grad = False


    def forward(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True)
        pooled_output = outputs[1]
        return pooled_output
    


class NewsNet2(nn.Module):
    def __init__(self, params, dtype=torch.float32):
        super().__init__()

        # BERT
        self.bert_base = BertLayer().to(dtype)
        if params['bert_trainable'] == False:
            for param in self.bert_base.parameters():
                param.requires_grad = False

        # Text Representation Layers
        self.text_hidden_layers = nn.ModuleList()
        for i in range(params['text_no_hidden_layer']):
            self.text_hidden_layers.append(nn.Linear(params['text_hidden_neurons'][i][0], params['text_hidden_neurons'][i][1], dtype=dtype))
            self.text_hidden_layers.append(nn.ReLU())
            self.text_hidden_layers.append(nn.Dropout(params['dropout']))

        final_len = params['text_hidden_neurons'][-1][1]
        self.text_repr = nn.Sequential(nn.Linear(final_len, params['repr_size'], dtype=dtype), nn.ReLU())

        # Image Representation Layers
        self.conv_base = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').to(dtype)
        self.conv_base = nn.Sequential(self.conv_base.features, nn.Flatten())
        if params['vgg_trainable'] == False:
            for param in self.conv_base.parameters():
                param.requires_grad = False
    
        self.image_hidden_layers = nn.ModuleList()
        for i in range(params['vis_no_hidden_layer']):
            self.image_hidden_layers.append(nn.Linear(params['vis_hidden_neurons'][i][0], params['vis_hidden_neurons'][i][1], dtype=dtype))
            self.image_hidden_layers.append(nn.ReLU())
            self.image_hidden_layers.append(nn.Dropout(params['dropout']))

        final_len = params['vis_hidden_neurons'][-1][1]
        self.image_repr = nn.Sequential(nn.Linear(final_len, params['repr_size'], dtype=dtype), nn.ReLU())

        # Combined Representation Layers
        self.combined_dropout = nn.Dropout(params['dropout'])
        self.combine_hidden_layers = nn.ModuleList()
        for i in range(params['final_no_hidden_layer']):
            self.combine_hidden_layers.append(nn.Linear(params['final_hidden_neurons'][i][0], params['final_hidden_neurons'][i][1], dtype=dtype))
            self.combine_hidden_layers.append(nn.ReLU())
            self.combine_hidden_layers.append(nn.Dropout(params['dropout']))
        
        final_len = params['final_hidden_neurons'][-1][1]
        self.prediction = nn.Sequential(nn.Linear(final_len, 1, dtype=dtype), nn.Sigmoid())


    def forward(self, input_texts, input_images):
        # BERT
        bert_output = self.bert_base(input_texts)

        # Text Representation
        for layer in self.text_hidden_layers:
            bert_output = layer(bert_output)
        text_repr = self.text_repr(bert_output)

        # Image Representation
        image_output = self.conv_base(input_images)
        for layer in self.image_hidden_layers:
            image_output = layer(image_output)
        image_repr = self.image_repr(image_output)

        # Combined Representation
        combined_repr = torch.cat([text_repr, image_repr], dim=1)
        combined_repr = self.combined_dropout(combined_repr)
        for layer in self.combine_hidden_layers:
            combined_repr = layer(combined_repr)

        # Prediction
        prediction = self.prediction(combined_repr)

        return prediction