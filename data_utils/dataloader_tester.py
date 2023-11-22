from data_loaders import *

dataset = NsElasticDataset("/home/ashiq/Desktop/CODANO/Data/NS_ES")

v,p,d = dataset.get_data(0.1, -0.5, -0.5, -0.1)

print(v.shape, v.dtype)
print(p.shape, p.dtype)
print(d.shape, d.dtype)

train, test = dataset.get_dataloader([0.1],2,)
normalizer = dataset.normalizer
for x,y in train:
    print(x.shape, y.shape)
    x, y = normalizer(x), normalizer(y)
    print(torch.max(x),torch.min(x), torch.max(y), torch.min(y))
    break