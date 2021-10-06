

from fastreid.utils.reid_patch import save_image
import torch

def SMA(q_loader, attack, model,pos,device='cuda'):
    
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
    model.eval()

    for _,data in enumerate(q_loader):
        with torch.no_grad():
            img=data['images'].to(device)
            raw_features = model(img/255.0)
        image_adv = attack.perturb(img/255.0,raw_features.to(device))

        path = data["img_paths"]
        save_image(image_adv,path,pos)
