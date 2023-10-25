import pickle
import torch
import random

def get_onestep_dataloader(train_test_split=0.2, location='../Data/MP_data/', batch_size=1):
    with open(location+'displacements0-5000.pkl', 'rb') as file:
        displacements = torch.tensor(pickle.load(file))
    with open(location+'pressures0-5000.pkl', 'rb') as file:
        # scaler variable
        pressure = torch.tensor(pickle.load(file))[:,:,None]
    with open(location+'velocitoes0-5000.pkl', 'rb') as file:
        velocities = torch.tensor(pickle.load(file))

    combined = torch.cat([displacements,velocities,pressure], dim= -1)
    step_t0 = combined[:-1, ...]
    step_t1 =  combined[1:, ...]

    indexs = [i for i in range(step_t0.shape[0])]
    ntrain = int(train_test_split*len(indexs))
    random.shuffle(indexs)
    train_t0,test_t0 = step_t0[indexs[:ntrain]], step_t0[indexs[ntrain:]]
    train_t1,test_t1 = step_t1[indexs[:ntrain]], step_t1[indexs[ntrain:]]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_t0, train_t1),\
                                           batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_t0, test_t1),\
                                           batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


        