import torch
from architectures import weights, fornet
from torch.utils.model_zoo import load_url

net_model = "EfficientNetAutoAttB4"
train_db = "DFDC"

model_url = weights.weight_url["{:s}_{:s}".format(net_model, train_db)]
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

net = getattr(fornet, net_model)().eval().to(device)
net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))
