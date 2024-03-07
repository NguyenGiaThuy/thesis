import torch
import torch.nn as nn
import torch.nn.functional as F



class NewsNet(nn.Module):
    def __init__(self,
                 headline_size, content_size, image_size,
                 hidden_len=32, output_len=200, embedding_len=300, num_filters=256,
                 filter_sizes=[3, 4], dropout_prob=0.5, alpha=0.6, beta=0.4, 
                 dtype=torch.float64):     
          
        super(NewsNet, self).__init__()
        
        self.dtype = dtype
        
        self.headline_size = headline_size
        self.content_size = content_size
        self.image_size = image_size
        
        self.hidden_len = hidden_len
        self.output_len = output_len
        self.embedding_len = embedding_len
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.total_num_filters = self.num_filters * len(self.filter_sizes)
        self.dropout_prob = dropout_prob
        
        self.alpha = torch.tensor(alpha, dtype=self.dtype)
        self.beta = torch.tensor(beta, dtype=self.dtype)
        
        # ----------Input transform layer----------
        self.headline_proj = nn.Linear(self.embedding_len, self.hidden_len, dtype=self.dtype)
        self.content_proj = nn.Linear(self.embedding_len, self.hidden_len, dtype=self.dtype)
        self.image_proj = nn.Linear(self.embedding_len, self.hidden_len, dtype=self.dtype)
            
        # ----------Headline convolutional layers----------
        # Text-CNN
        self.headline_conv_layers = nn.ModuleList()
        for filter_size in self.filter_sizes:
            self.headline_conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=1, 
                          out_channels=self.num_filters, 
                          kernel_size=(filter_size, self.hidden_len), 
                          dtype=self.dtype),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(self.headline_size - filter_size + 1, 1)),
            ))
        
        # ----------Dropout and feedforward layer----------
        self.headline_fc = nn.Sequential(
            nn.Linear(self.total_num_filters, self.output_len, dtype=self.dtype),
            nn.ReLU(),
            nn.LayerNorm(self.output_len, dtype=dtype) 
        )
            
        # ----------Content convolutional layers----------
        # Text-CNN
        self.content_conv_layers = nn.ModuleList()
        for filter_size in self.filter_sizes:
            self.content_conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=1, 
                          out_channels=self.num_filters, 
                          kernel_size=(filter_size, self.hidden_len), 
                          dtype=self.dtype),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(self.content_size - filter_size + 1, 1)),
            ))
        
       # ----------Dropout and feedforward layer----------
        self.content_fc = nn.Sequential(
            nn.Linear(self.total_num_filters, self.output_len, dtype=self.dtype),
            nn.ReLU(),
            nn.LayerNorm(self.output_len, dtype=dtype) 
        )  
        
        # ----------Image convolutional layers----------
        # Text-CNN
        self.image_conv_layers = nn.ModuleList()
        for filter_size in self.filter_sizes:
            self.image_conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=1, 
                          out_channels=self.num_filters, 
                          kernel_size=(filter_size, self.hidden_len), 
                          dtype=self.dtype),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(self.image_size - filter_size + 1, 1)),   
            ))
        
        # ----------Dropout and feedforward layer----------
        self.image_fc = nn.Sequential(
            nn.Linear(self.total_num_filters, self.output_len, dtype=self.dtype),
            nn.ReLU(),
            nn.LayerNorm(self.output_len, dtype=dtype) 
        )
        
        # ----------Combined feedforward layer----------
        self.combined_fc = nn.Sequential(
            nn.Linear(self.output_len * 3, 2, dtype=self.dtype),
            nn.Softmax(dim=1)
        )
         
    def forward(self, input_x): 
        if input_x == None:
            self.batch_size = 64
            input_headline_ = torch.empty([self.batch_size, self.headline_size, self.embedding_len], dtype=self.dtype)
            input_content_ = torch.empty([self.batch_size, self.content_size, self.embedding_len], dtype=self.dtype)
            input_image_ = torch.empty([self.batch_size, self.image_size, self.embedding_len], dtype=self.dtype)
        else:
            self.batch_size = len(input_x[0])
            input_headline_, input_content_, input_image_ = input_x
        
        # ----------Input transform layer----------
        self.input_headline = self.headline_proj(input_headline_)
        self.input_content = self.content_proj(input_content_)
        self.input_image = self.image_proj(input_image_)
        
        # ----------Headline convolutional layers----------
        # Pass through text-CNN
        pooled_headline_outputs = []
        for conv in self.headline_conv_layers:
            temp = self.input_headline.view(self.batch_size, 1, self.headline_size, self.hidden_len)
            pooled_headline_conv = conv(temp)
            pooled_headline_outputs.append(pooled_headline_conv)
    
        # Ravel headline output
        concat_pooled_headline = torch.cat(pooled_headline_outputs, dim=1)
        flat_pooled_headline = concat_pooled_headline.view(-1, self.total_num_filters)
        
        # ----------Dropout and feedforward layer----------
        headline_vector = self.headline_fc(flat_pooled_headline)
        
        # ----------Content convolutional layers----------
        # Pass through text-CNN
        pooled_content_outputs = []
        for conv in self.content_conv_layers:
            temp = self.input_content.view(self.batch_size, 1, self.content_size, self.hidden_len)
            pooled_content_conv = conv(temp)
            pooled_content_outputs.append(pooled_content_conv)

        # Ravel content output
        concat_pooled_content = torch.cat(pooled_content_outputs, dim=1)
        flat_pooled_content = concat_pooled_content.view(-1, self.total_num_filters)

        # ----------Dropout and feedforward layer----------
        content_vector = self.content_fc(flat_pooled_content)
        
        # ----------Image convolutional layers----------
        # Pass through text-CNN
        pooled_image_outputs = []
        for conv in self.image_conv_layers:
            temp = self.input_image.view(self.batch_size, 1, self.image_size, self.hidden_len)
            pooled_image_conv = conv(temp)
            pooled_image_outputs.append(pooled_image_conv)

        # Ravel image output
        concat_pooled_image = torch.cat(pooled_image_outputs, dim=1)
        flat_pooled_image = concat_pooled_image.view(-1, self.total_num_filters)

        # ----------Dropout and feedforward layer----------
        image_vector = self.image_fc(flat_pooled_image)
        
        # ----------Vector combination----------
        self.combined_text = torch.cat([headline_vector, content_vector], dim=1)
        self.combined_image = torch.cat([image_vector, image_vector], dim=1)
        concat_vector = torch.cat([headline_vector, content_vector, image_vector], dim=1)
        self.combined_vector = self.combined_fc(concat_vector)
        
        # ----------Prediction----------
        self.pred_y = self.combined_vector
        
        return self.pred_y
    
    
    def cross_modal_loss(self, pred_y, input_y): 
        if input_y == None:
            self.input_y = torch.empty([self.batch_size, 2], dtype=self.dtype)
        else:
            self.input_y = F.one_hot(input_y, num_classes=2).to(self.dtype)
        
        # ----------Cosine similarity----------
        cos_sim = (1 + F.cosine_similarity(self.combined_text, self.combined_image, dim=1)) / 2
        distance = torch.ones_like(cos_sim, dtype=self.dtype) - cos_sim
        self.cos = torch.stack([distance, cos_sim], dim=1)
         
        # ----------Loss----------
        self.loss1 = F.cross_entropy(self.input_y , pred_y)
        self.loss2 = F.cross_entropy(self.input_y , self.cos)

        self.loss = self.alpha * self.loss1 + self.beta * self.loss2
        return self.loss


    