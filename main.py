import os
import time
import torch
import pandas as pd
import torch.optim as optim
from utils import load_configs
from test import test_model3
from train import train_model3
from models import news_net3
from torch.utils.data import DataLoader
from utils.visualize_data import draw_linechart, draw_confusion_matrix, count_labels
from utils.transform_data import TransformerDataset, custom_collate_fn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import BertTokenizer as TextTokenizer
from torch.optim.lr_scheduler import StepLR
from models.news_net3 import AdversarialLoss
import sys



if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available on this system.")
    else:
        device = 'cpu'
        print("CUDA is not available on this system.")
        
    arguments = sys.argv

    # ----------Load configs----------
    dataset_config, model_config, train_eval_config = load_configs.load()
    train_dir = os.path.join(os.getcwd(), dataset_config['train_dir'])
    dev_dir = os.path.join(os.getcwd(), dataset_config['dev_dir'])
    test_dir = os.path.join(os.getcwd(), dataset_config['test_dir'])
    images_dir = os.path.join(os.getcwd(), dataset_config['images_dir'])
    
    # ----------Initialize model----------
    dtype = torch.float32
    model_params = {
        'text_encoder': model_config['text_encoder'],
        'text_encoder_trainable': model_config['text_encoder_trainable'],
        'n_trainable_layers': model_config['n_trainable_layers'],
        # 'image_encoder': model_config['image_encoder'][0], # VGG19
        'image_encoder': model_config['image_encoder'][1], # ResNet50
        'image_encoder_trainable': model_config['image_encoder_trainable'],
        'dropout_prob': model_config['dropout_prob'],
        'repr_size':  model_config['repr_size'],
        'lambd': model_config['lambd']
    }
    
    tokenizer = TextTokenizer.from_pretrained(model_params['text_encoder'])
    model = news_net3.NewsNet3(model_params).to(dtype).to(device)
    criterion = AdversarialLoss(train_eval_config['alpha']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_eval_config['lr'], weight_decay=train_eval_config['l2_reg_lambd'])
    scheduler = StepLR(optimizer, step_size=train_eval_config['lr_step_size'])
    print(f'total model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # ----------Train model----------
    if len(arguments) >= 2 and arguments[1] == 'train':
        # Load and transform data
        print('loading data...')
        train_data = TransformerDataset(train_dir, images_dir, model_params['image_encoder'])
        dev_data = TransformerDataset(dev_dir, images_dir, model_params['image_encoder'])
        num_train_labels = count_labels(train_data)
        total_train_labels = num_train_labels[0] + num_train_labels[1]
        num_dev_labels = count_labels(dev_data)
        total_dev_labels = num_dev_labels[0] + num_dev_labels[1]
        
        print(f'train: {num_train_labels[0]}/{total_train_labels} {num_train_labels[1]}/{total_train_labels}')
        print(f'dev: {num_dev_labels[0]}/{total_dev_labels} {num_dev_labels[1]}/{total_dev_labels}')
        print('loading data done')

        checkpoint_dir = None
        if len(arguments) == 3 and (arguments[2] != None or arguments[2] != ''):
            checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints', arguments[2])

        # Train model
        print("training model...")
        start_time = time.time()

        train_loader = DataLoader(train_data, batch_size=train_eval_config['batch_size'], 
                                    shuffle=True, collate_fn=custom_collate_fn, num_workers=4)
        dev_loader = DataLoader(dev_data, batch_size=train_eval_config['batch_size'], 
                                shuffle=False, collate_fn=custom_collate_fn, num_workers=4)
        train_losses, dev_losses, train_accuracies, dev_accuracies, best_model = train_model3(model=model,
                                                                                                criterion=criterion,
                                                                                                optimizer=optimizer,
                                                                                                scheduler=scheduler,
                                                                                                train_loader=train_loader, 
                                                                                                dev_loader=dev_loader,
                                                                                                tokenizer=tokenizer,
                                                                                                num_epochs=train_eval_config['num_epochs'],
                                                                                                warm_up=train_eval_config['warm_up'],
                                                                                                tolerance=train_eval_config['tolerance'], 
                                                                                                checkpoint_dir=checkpoint_dir,
                                                                                                device=device, dtype=dtype)
            
        avg_train_accuracy = torch.mean(torch.tensor(train_accuracies, dtype=dtype), dtype=dtype)
        avg_dev_accuracy = torch.mean(torch.tensor(dev_accuracies, dtype=dtype), dtype=dtype)

        # Plot and save train and dev losses and accuracies
        losses = {'epoch': list(range(1, len(train_losses) + 1)) + list(range(1, len(dev_losses) + 1)),
                  'loss': train_losses + dev_losses,
                  'legend': ['train loss'] * len(train_losses) + ['dev loss'] * len(dev_losses)}
        losses_df = pd.DataFrame(losses)
        draw_linechart(losses_df, 'loss', f'train_dev_losses')

        accuracies = {'epoch': list(range(1, len(train_accuracies) + 1)) + list(range(1, len(dev_accuracies) + 1)),
                      'accuracy': train_accuracies + dev_accuracies,
                      'legend': ['train accuracy'] * len(train_accuracies) + ['dev accuracy'] * len(dev_accuracies)}
        accuracies_df = pd.DataFrame(accuracies)
        draw_linechart(accuracies_df, 'accuracy', f'train_dev_accuracies')

        torch.save(best_model['model_state_dict'], 'best_model.pth')
            
        print(f'\ntrain duration: {time.time() - start_time:.0f} seconds')
        print('training model done')
        
    # ----------Test model----------
    elif len(arguments) == 3 and arguments[1] == 'test' and (arguments[2] != None or arguments[2] != ''):
        # Load and transform data
        print('loading data...')
        test_data = TransformerDataset(test_dir, images_dir, model_params['image_encoder'])
        num_test_labels = count_labels(test_data)
        total_test_labels = num_test_labels[0] + num_test_labels[1]
        print(f'test: {num_test_labels[0]}/{total_test_labels} {num_test_labels[1]}/{total_test_labels}')
        print('loading data done')

        model_dir = os.path.join(os.getcwd(), arguments[2])
        model.load_state_dict(torch.load(model_dir))
        
        # Test model
        print('\ntesting model...')
        start_time = time.time()
        test_loader = DataLoader(test_data, batch_size=train_eval_config['batch_size'], 
                                 shuffle=False, collate_fn=custom_collate_fn, num_workers=4)
        predictions, targets = test_model3(model=model, 
                                           criterion=criterion,
                                           test_loader=test_loader, 
                                           tokenizer=tokenizer,
                                           device=device,
                                           dtype=dtype)

        print(f'\nduration: {time.time() - start_time:.0f} seconds')
        print('testing model done')

        # Evaluate metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        f1 = f1_score(targets, predictions)
        print(f'test accuracy: {accuracy: 8.6f}')
        print(f'test precision: {precision: 8.6f}')
        print(f'test recall: {recall: 8.6f}')
        print(f'test f1: {f1: 8.6f}')

        # Draw confusion matrix
        draw_confusion_matrix(predictions, targets, ['fake', 'real'], 'test')