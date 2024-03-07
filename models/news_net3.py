import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import torchvision.models as models
from torch.autograd import Function



class TextEncoderLayer(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        self.text_encoder = BertModel.from_pretrained(params['text_encoder'])
        
        for name, param in self.text_encoder.named_parameters():
            if 'encoder.layer.' in name and int(name.split('.')[2]) < self.text_encoder.config.num_hidden_layers - params['n_trainable_layers']:
                param.requires_grad = False
                
        if params['text_encoder_trainable'] == False:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.text_encoder(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True)
        sentence_repr = outputs.pooler_output
        word_repr = outputs.last_hidden_state
        return sentence_repr, word_repr
    
    
class ImageEncoderLayer(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        hidden_size = {'vgg19': 25088, 'resnet50': 2048}
        
        self.image_encoder = models.vgg19(weights='DEFAULT')
        if params['image_encoder'] == 'resnet50':  
            self.image_encoder = models.resnet50(weights='DEFAULT')
            
        if params['image_encoder_trainable'] == False:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
                
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1], 
                                           nn.Flatten(), nn.Linear(hidden_size[params['image_encoder']], 768)) 

    def forward(self, inputs):
        outputs = self.image_encoder(inputs)
        return outputs


class MLP(nn.Module):    
    def __init__(self, hidden_sizes, dropout):
        super().__init__()
        
        self.mlp = nn.ModuleList()
        for hidden_size in hidden_sizes:
            self.mlp.append(nn.Linear(hidden_size[0], hidden_size[1]))
            self.mlp.append(nn.ELU(inplace=True))
            self.mlp.append(nn.Dropout(dropout))    
    
    def forward(self, inputs):
        outputs = inputs
        for module in self.mlp:
            outputs = module(outputs)
        return outputs


class CoAttentionLayer(nn.Module):
    def __init__(self, query_size, key_size, attention_size):
        super().__init__()
        
        self.w_relation = nn.Parameter(torch.zeros((query_size, key_size)))
        self.W_q = nn.Parameter(torch.zeros(query_size, attention_size))
        self.W_k = nn.Parameter(torch.zeros(key_size, attention_size))
        self.W_q_att =  nn.Parameter(torch.zeros(attention_size, 1))
        self.W_k_att =  nn.Parameter(torch.zeros(attention_size, 1))

    def forward(self, queries, keys):
        relation_scores = torch.tanh(torch.bmm(queries, torch.matmul(self.w_relation, torch.transpose(keys, 1, 2))))
        
        query_proj = torch.matmul(queries, self.W_q)
        key_proj = torch.matmul(keys, self.W_k)
        
        query_att_score = torch.tanh(query_proj + torch.bmm(relation_scores, key_proj))
        key_att_score = torch.tanh(key_proj + torch.bmm(torch.transpose(relation_scores, 1, 2), query_proj))
        
        query_att_weights = F.softmax(torch.matmul(query_att_score, self.W_q_att), dim=1)
        key_att_weights = F.softmax(torch.matmul(key_att_score, self.W_k_att), dim=1)
           
        # weighted_queries = torch.bmm(torch.transpose(query_att_weights, 1, 2), queries)
        # weighted_keys = torch.bmm(torch.transpose(key_att_weights, 1, 2), keys)
        weighted_queries = torch.mul(query_att_weights, queries)
        weighted_keys = torch.mul(key_att_weights, keys)

        return weighted_queries, weighted_keys, query_att_weights, key_att_weights


class GradientReverse(Function):
    lambd = 1.0
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.lambd * grad_output.neg()
    
def grad_reverse(x, lambd=1.0):
    GradientReverse.lambd = lambd
    return GradientReverse.apply(x)


class NewsNet3(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        
        self.text_encoder = TextEncoderLayer(self.params)
        
        self.image_encoder = ImageEncoderLayer(self.params)

        # Co-attention between headline and content (content focused) - output shape (N, 50, 768), (N, 1, 768)
        self.headline_image_att1 = CoAttentionLayer(768, 768, self.params['repr_size'] * 2) 
        
        # Co-attention between content and image (image focused) - output shape (N, 1, 768), (N, 1, 768)
        self.headline_image_att2 = CoAttentionLayer(768, 768, self.params['repr_size'] * 2)
        
        self.headline_norm = nn.LayerNorm(768)
        self.image_norm = nn.LayerNorm(768)
              
        # Combined representation layers
        self.combined_repr_layers = MLP([(768 * 2, self.params['repr_size'] * 2)], self.params['dropout_prob'])
        
        self.classifier = nn.Sequential(nn.Linear(self.params['repr_size'] * 2, 1), nn.Sigmoid())
        
        self.domain_classifier = nn.Sequential(MLP([(self.params['repr_size'] * 2, self.params['repr_size'])], self.params['dropout_prob']), 
                                               nn.Linear(self.params['repr_size'], 22), nn.Softmax(dim=1))
        
    def forward(self, inputs):
        headline_inputs, image_inputs = inputs
        headline_inputs = (headline_inputs.input_ids, headline_inputs.attention_mask, headline_inputs.token_type_ids)
        
        headline_sentence_repr, headline_word_repr =self.text_encoder(headline_inputs)
        headline_sentence_repr = torch.unsqueeze(headline_sentence_repr, 1)
        image_repr = torch.unsqueeze(self.image_encoder(image_inputs), 1)
        
        headline_att1, image_att1, _, _ = self.headline_image_att1(headline_sentence_repr, image_repr)
        headline_att2, image_att2, _, _ = self.headline_image_att2(headline_word_repr, image_repr)
        
        headline_att1 = torch.squeeze(headline_att1, 1)
        headline_att2 = torch.mean(headline_att2, 1) # Reduce from (N, 50, 768) to (N, 768)
        image_att1 = torch.squeeze(image_att1, 1)
        image_att2 = torch.squeeze(image_att2, 1)
        
        headline_att = self.headline_norm(headline_att1 + headline_att2)
        image_att = self.image_norm(image_att1 + image_att2)
               
        combined_repr = torch.cat([headline_att, image_att], 1)
        combined_repr = self.combined_repr_layers(combined_repr)
        
        class_outputs = self.classifier(combined_repr)
        
        if self.training == True:
            reverse_feat = grad_reverse(combined_repr, self.params['lambd'])
            domain_outputs = self.domain_classifier(reverse_feat)
            
            return class_outputs, domain_outputs
        else:
            return class_outputs

    
class AdversarialLoss(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        
        self.alpha = alpha

    def forward(self, pred_class_labels, true_class_labels, pred_domain_labels=None, true_domain_labels=None):
        class_loss = F.binary_cross_entropy(pred_class_labels, true_class_labels)
        
        if pred_domain_labels != None and true_domain_labels != None:
            domain_loss = F.cross_entropy(pred_domain_labels, true_domain_labels)
            loss = class_loss + self.alpha * domain_loss
        
            return loss, class_loss, domain_loss
        else:
            return class_loss