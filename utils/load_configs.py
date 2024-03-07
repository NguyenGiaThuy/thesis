import os
import json



def load(dataset_config_json='dataset_config.json', 
         model_config_json='model_config.json', 
         train_eval_config_json='train_eval_config.json'):
    
    dataset_config_dir = os.path.join(os.getcwd(), 'configs', dataset_config_json)
    model_config_dir = os.path.join(os.getcwd(), 'configs', model_config_json)
    train_eval_config_dir = os.path.join(os.getcwd(), 'configs', train_eval_config_json)
    
    with open(dataset_config_dir) as dataset_config_file:
        dataset_config = json.load(dataset_config_file)
        
    with open(model_config_dir) as model_config_file:
        model_config = json.load(model_config_file)
        
    with open(train_eval_config_dir) as train_eval_config_file:
        train_eval_config = json.load(train_eval_config_file)
        
    return dataset_config, model_config, train_eval_config