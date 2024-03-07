import torch
import torch.nn.functional as F



def test_model3(model, criterion, test_loader, tokenizer, device, dtype=torch.float32):
    headline_len = 50
    total_predictions = []
    total_targets = []
    model.eval()
    with torch.no_grad():
        for b, (X_data, y_data) in enumerate(test_loader):
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
            print(f'batch: {b + 1}/{len(test_loader)} | ' +
                  f'loss: {class_loss.item(): 8.6f} | accuracy: {accuracy.item(): 8.6f}')

            # Log losses and accuracies
            total_predictions += (pred_y > 0.5).to(dtype).tolist()
            total_targets += input_y.tolist()
        
    return total_predictions, total_targets