import sys, os
sys.path.insert(0, '../../src/')
sys.path.insert(0, '../../data/geospatial_data/')
import argparse
import wandb
import dill as pkl
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from data_generator import train_val_test_split
from models import LinearSCI
from trainers import Trainer
from spatial_dataset_real import DurhamDataset

# Experimental tracking
wandb.login()

np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
torch.cuda.manual_seed_all(2023)
torch.backends.cudnn.deterministic = True


def main(args):
    # Load and re-format data
    with open('../../data/geospatial_data/pickle_files/durham_real_train.pkl', 'rb') as f:
        data = pkl.load(f)
    interventions = [np.array([x[3] for x in data]), np.array([x[4] for x in data])]
    confounder = np.array([x[2].mean(axis=(0,1)) for x in data])
    spatial_features = np.array([[x[0],x[1]] for x in data])
    targets = np.array([x[5] for x in data])
    scaler = StandardScaler()
    targets = scaler.fit_transform(targets.reshape(-1,1))

    train_dataset, val_dataset, test_dataset = train_val_test_split(
        interventions, confounder, spatial_features, targets, 
        train_size=0.7, val_size=0.3, test_size=0.0, shuffle=False, 
        block_sampling=True
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    
    # Model definition
    model = LinearSCI(
        num_interventions=len(interventions), 
        window_size=interventions[0].shape[-1],  
        confounder_dim=confounder.shape[-1], 
        unobserved_confounder=True, 
        dimensionality=2, 
        kernel_func="rbf", 
        kernel_param_vals=[1.,100.,0.1], 
        inducing_point_space=[[0.,2253.],[0.,2307.]]
    )
    
    # Experimental tracking
    wandb.init(
        project="deep_sci",
        name="geospatial_real_linear_with_U", 
        allow_val_change=True
    )
    config = wandb.config
    
    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = args.optim_name
    optim_params = {
        'lr': args.lr,
        'momentum': args.momentum
    }
    epochs, patience = args.n_epochs, args.patience
    trainer = Trainer(
        model=model, 
        data_generators=dataloaders, 
        optim=optim, 
        optim_params=optim_params, 
        device=device,
        epochs=epochs,
        patience=patience, 
        wandb=wandb
    )
    trainer.train()
    
    # Inference
    with open('../../data/geospatial_data/pickle_files/durham_real_test.pkl', 'rb') as f:
        test_data = pkl.load(f)
        data_generator = test_data['data_generator']
        height, width = test_data['height'], test_data['width']
        
    window_size = interventions[0].shape[-1]
    padding_h, padding_w = (height+1-1500)//2, (width+1-1500)//2
    de_map, ie_map, te_map = [], [], []
    with torch.no_grad():
        for i in tqdm(range(padding_h, height-padding_h+1),position=0,leave=True):
            batch = [[torch.empty(0).to(device)]*len(interventions),torch.empty(0).to(device),torch.empty(0).to(device)]
            for j in range(padding_w, width-padding_w+1):
                sample = data_generator.get_item_by_coords(i,j)
                s = torch.tensor([i,j]).float().view(1,-1).to(device)
                nlcd = torch.tensor(sample[2]).mean(dim=(0,1)).float().view(1,-1).to(device)
                ndvi = torch.tensor(sample[3]).float().view(1,sample[3].shape[0],-1).to(device)
                albedo = torch.tensor(sample[4]).float().view(1,sample[3].shape[0],-1).to(device)
                batch[0][0] = torch.cat((batch[0][0],ndvi),dim=0)
                batch[0][1] = torch.cat((batch[0][1],albedo),dim=0)
                batch[1] = torch.cat((batch[1],nlcd),dim=0)
                batch[2] = torch.cat((batch[2],s),dim=0)
                del sample, s, nlcd, ndvi, albedo
            y_pred_11 = model.predict(*batch).cpu().numpy()
            y_pred_11 = scaler.inverse_transform(y_pred_11.reshape(-1,1)).squeeze()
            for i in range(len(batch[0])):
                batch[0][i][:,window_size//2,window_size//2] = 0.
            y_pred_01 = model.predict(*batch).cpu().numpy()
            y_pred_01 = scaler.inverse_transform(y_pred_01.reshape(-1,1)).squeeze()
            for i in range(len(batch[0])):
                batch[0][i] = torch.zeros_like(batch[0][i])
            y_pred_00 = model.predict(*batch).cpu().numpy()
            y_pred_00 = scaler.inverse_transform(y_pred_00.reshape(-1,1)).squeeze()
            de_map.append(y_pred_11 - y_pred_01)
            ie_map.append(y_pred_01 - y_pred_00)
            te_map.append(y_pred_11 - y_pred_00)
    
    de_map, ie_map, te_map = np.array(de_map), np.array(ie_map), np.array(te_map)
    os.makedirs('./results', exist_ok=True)
    result = {'de': de_map, 'ie': ie_map, 'te': te_map}
    with open('./results/results_linear_with_U.pkl', 'wb') as f:
        pkl.dump(result, f)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Linear, with unobserved confounder')
    arg_parser.add_argument('--batch_size', type=int, default=128)
    arg_parser.add_argument('--optim_name', type=str, default="sgd")
    arg_parser.add_argument('--lr', type=float, default=1e-5)
    arg_parser.add_argument('--momentum', type=float, default=0.99)
    arg_parser.add_argument('--weight_decay', type=float, default=0.0)
    arg_parser.add_argument('--n_epochs', type=int, default=1000)
    arg_parser.add_argument('--patience', type=int, default=20)
    args = arg_parser.parse_known_args()[0]
    main(args)