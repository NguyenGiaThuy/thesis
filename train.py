import torch
import copy
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR



def train_model3(model, criterion, optimizer, 
                 scheduler, train_loader, dev_loader, 
                 tokenizer, num_epochs, warm_up, tolerance, device, 
                 checkpoint_dir=None, dtype=torch.float32):
    headline_len = 50
    checkpoint = {'epoch': 0,
                  'train_losses': None,
                  'dev_losses': None,
                  'train_accuracies': None,
                  'dev_accuracies': None,
                  'model_state_dict': None,
                  'criterion_state_dict': None, 
                  'optimizer_state_dict:': None,
                  'scheduler_state_dict': None}
    best_model = {'loss': float('inf'), 'accuracy': 0.0, 'model_state_dict': None}
    
    # Load checkpoint (if applicable)
    start_epoch = 0
    train_losses = []
    dev_losses = []
    train_accuracies = []
    dev_accuracies = []
    if checkpoint_dir != None:
        checkpoint = torch.load(checkpoint_dir, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        dev_losses = checkpoint['dev_losses']
        train_accuracies = checkpoint['train_accuracies']
        dev_accuracies = checkpoint['dev_accuracies']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        criterion.load_state_dict(checkpoint['criterion_state_dict'])
    
    # Train and evaluate
    cur_tolerance = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        batch_train_losses = []
        batch_train_accuracies = []
        for b, (X_data, y_data) in enumerate(train_loader):
            texts, images = X_data
            
            tokenized_headlines = tokenizer.batch_encode_plus(
                texts,
                add_special_tokens=True,
                max_length=headline_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            ).to(device)
            
            input_images = torch.cat(images).to(dtype).to(device)
            inputs_X = (tokenized_headlines, input_images)
            input_y = torch.squeeze(y_data[0]).to(dtype).to(device)
            input_domain = torch.squeeze(y_data[1])
            input_domain = F.one_hot(input_domain, 22).to(dtype).to(device)
            
            # Apply the model
            optimizer.zero_grad()
            pred_y, pred_domain = model(inputs_X)
            pred_y = torch.squeeze(pred_y)
            pred_domain = torch.squeeze(pred_domain)
            loss, class_loss, _ = criterion(pred_y, input_y, pred_domain, input_domain)

            # Evaluate accuracy
            predictions = torch.eq((pred_y > 0.5), input_y).to(dtype)
            accuracy = torch.mean(predictions, dtype=dtype)
            
            # Update model
            loss.backward()
            optimizer.step()
            
            # Print results by batches
            if (b + 1) % max(len(train_loader) // 5, 1) == 0 or (b + 1) == len(train_loader):
                print(f'epoch: {epoch + 1}/{num_epochs} | batch: {b + 1}/{len(train_loader)} | ' +
                      f'train loss: {class_loss.item(): 8.6f} | train accuracy: {accuracy.item(): 8.6f}')
            
            batch_train_losses.append(class_loss.item())
            batch_train_accuracies.append(accuracy.item())

        # Log train losses and train accuracies
        avg_train_loss = torch.mean(torch.tensor(batch_train_losses, dtype=dtype), dtype=dtype)
        train_losses.append(avg_train_loss.item())
        
        avg_train_accuracy = torch.mean(torch.tensor(batch_train_accuracies, dtype=dtype), dtype=dtype)
        train_accuracies.append(avg_train_accuracy.item())
        
        model.eval()
        with torch.no_grad():
            batch_dev_losses = []
            batch_dev_accuracies = []
            for b, (X_data, y_data) in enumerate(dev_loader):
                texts, images = X_data
                
                tokenized_headlines = tokenizer.batch_encode_plus(
                    texts,
                    add_special_tokens=True,
                    max_length=headline_len,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    return_tensors='pt'
                ).to(device)
         
                input_images = torch.cat(images).to(dtype).to(device)
                inputs_X = (tokenized_headlines, input_images)
                input_y = torch.squeeze(y_data[0]).to(dtype).to(device)

                # Apply the model
                pred_y = model(inputs_X)
                pred_y = torch.squeeze(pred_y)
                class_loss = criterion(pred_y, input_y)

                # Evaluate accuracy
                predictions = torch.eq((pred_y > 0.5), input_y).to(dtype)
                accuracy = torch.mean(predictions, dtype=dtype)
                                
                # Print results by batches
                if (b + 1) % max(len(dev_loader) // 5, 1) == 0 or (b + 1) == len(dev_loader):
                    print(f'epoch: {epoch + 1}/{num_epochs} | batch: {b + 1}/{len(dev_loader)} | ' +
                          f'dev loss: {class_loss.item(): 8.6f} | dev accuracy: {accuracy.item(): 8.6f}')
                
                batch_dev_losses.append(class_loss.item())
                batch_dev_accuracies.append(accuracy.item())
                        
        # Log train losses and train accuracies
        avg_dev_loss = torch.mean(torch.tensor(batch_dev_losses, dtype=dtype), dtype=dtype)
        dev_losses.append(avg_dev_loss.item())
        
        avg_dev_accuracy = torch.mean(torch.tensor(batch_dev_accuracies, dtype=dtype), dtype=dtype)
        dev_accuracies.append(avg_dev_accuracy.item())
        
        print(f'-----epoch: {epoch + 1}/{num_epochs} | ' +
              f'train loss: {train_losses[-1]: 8.6f} | train accuracy: {train_accuracies[-1]: 8.6f} | ' +
              f'dev loss: {dev_losses[-1]: 8.6f} | dev accuracy: {dev_accuracies[-1]: 8.6f}-----')
        
        scheduler.step()
        
        # Save checkpoints
        if (epoch + 1) % 5 == 0:
            checkpoint['epoch'] = epoch
            checkpoint['train_losses'] = copy.deepcopy(train_losses)
            checkpoint['dev_losses'] = copy.deepcopy(dev_losses)
            checkpoint['train_accuracies'] = copy.deepcopy(train_accuracies)
            checkpoint['dev_accuracies'] = copy.deepcopy(dev_accuracies)
            checkpoint['model_state_dict'] = copy.deepcopy(model.state_dict())
            checkpoint['criterion_state_dict'] = copy.deepcopy(criterion.state_dict())
            checkpoint['optimizer_state_dict'] = copy.deepcopy(optimizer.state_dict())
            checkpoint['scheduler_state_dict'] = copy.deepcopy(scheduler.state_dict())
                    
            checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints', 'checkpoint_' + str(epoch + 1) + '.pth')
            torch.save(checkpoint, checkpoint_dir)
        
        # Early stopping and save the best model
        if epoch >= warm_up: 
            if dev_accuracies[-1] < best_model['accuracy']:
                cur_tolerance += 1
            else:
                best_model['loss'] = dev_losses[-1]
                best_model['accuracy'] = dev_accuracies[-1]
                best_model['model_state_dict'] = copy.deepcopy(model.state_dict())
                cur_tolerance = 0
        if cur_tolerance >= tolerance:
            break
                    
    return train_losses, dev_losses, train_accuracies, dev_accuracies, best_model

