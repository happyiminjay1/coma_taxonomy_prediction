import torch

score = torch.tensor( [0.8982,
                       0.805,
                       0.6393,
                       0.9983,
                       0.5731,
                       0.0469,
                       0.556,
                       0.1476,
                       0.8404,
                       0.5544] )

target = torch.tensor( 10 )

criterion = torch.nn.CrossEntropyLoss()
loss = criterion(score,target)

print(loss.item())