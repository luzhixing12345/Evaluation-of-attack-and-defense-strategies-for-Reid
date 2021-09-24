

import torch

def SMA(g_loader, attack, model, device='cuda'):
    
    """Perturb the gallery with SMA
    
    Arguments:
        g_loader {pytorch dataloader} -- dataloader of the gallery dataset
        attack {advertorch.attack} -- adversarial attack to perform on the gallery
        model {pytorch model} -- pytorch model to evaluate
        device {cuda device} --
    
    Returns:
        features -- Tensor of the features of the gallery
        ids -- numpy array of ids
        cams -- numpy array of camera ids
    """
    ids = []
    cams = []
    features = []
    model.eval()

    for _,data in enumerate(g_loader):
        with torch.no_grad():
            img=data['image'].to(device)
            raw_features = model(img)
        
        image_adv = attack.perturb(img, raw_features.to(device))
        with torch.no_grad():
            output = model(image_adv.to(device))
        features.append(output)
        ids.append(data['targets'].cpu())
        cams.append(data['camid'].cpu())

    ids = torch.cat(ids, 0)
    cams = torch.cat(cams, 0)
    features = torch.cat(features, 0)

    return features.cpu(), ids.numpy(), cams.numpy()