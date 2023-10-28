import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from timeit import default_timer
import gc
from tqdm import tqdm
def simple_trainer(model,train_loader, test_loader, params):
    lr = params.lr
    weight_decay = params.weight_decay
    scheduler_step = params.scheduler_step
    scheduler_gamma = params.scheduler_gamma
    epochs = params.epochs
    weigh_path = params.weight_path
    optimizer = Adam(model.parameters(), lr=lr, \
                     weight_decay=weight_decay,amsgrad = False)
    scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    loss_p = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_count = 0
        train_loader_iter = tqdm(train_loader, desc=f'Epoch {ep}/{epochs}', leave=False, ncols=100)
        for x, y in train_loader_iter:
            x, y = x.cuda(), y.cuda()
            batch_size = x.shape[0]
            optimizer.zero_grad()
            out, _, _, _ = model(x)
            train_count += 1

            loss = loss_p(out.reshape(batch_size,-1), y.reshape(batch_size,-1))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()
            del x,y,out,loss
            gc.collect()
        torch.cuda.empty_cache()
        scheduler.step()
        t2 = default_timer()
        print(f"Epoch {ep}: Time: {t2 - t1:.2f}s, Loss: {train_l2 / train_count:.4f}")

    #torch.save(model.state_dict(), weigh_path)
    
    model.eval()
    test_l2 = 0.0
    ntest = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            batch_size = x.shape[0]
            out,_,_,_ = model(x)
            ntest +=1
            test_l2 += loss_p(out.reshape(batch_size,-1), y.reshape(batch_size,-1)).item()

    test_l2 /= ntest

    t2 = default_timer()
    print("Test Error : ", test_l2)