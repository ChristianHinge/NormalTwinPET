import torch
import numpy as np
import scipy
import torch
import torchio as tio
from abc import abstractmethod
from tqdm import tqdm

class StitcherBase:
    def __init__(self, patch_overlap, patch_size, batch_size, device="cpu"):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.patch_overlap = patch_overlap
        self.patch_size = patch_size

    def _get_grid_sampler(self,subject):
        return tio.inference.GridSampler(
        subject,
        self.patch_size,
        self.patch_overlap,
        )
    def __call__(self, model,batch):
        sub = tio.Subject(batch)
        #sub, sh = self._pad(sub)
        grid_sampler = self._get_grid_sampler(sub)

        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size)
        with torch.no_grad():
            pred = self.forward(model,patch_loader,grid_sampler)
        #pred = self._unpad(sub,pred, sh)
        return pred.cpu()

    def recursive_apply(self, batch):
        if isinstance(batch, dict):
            return {k: self.recursive_apply(v) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, list):
            return [self.recursive_apply(x) for x in batch]
        elif isinstance(batch, tuple):
            return tuple([self.recursive_apply(x) for x in batch])
        else:
            return batch

    def move_dict_to_device(self, d):
        for key in d.keys():
            if isinstance(d[key], dict) and "data" in d[key]:
                if isinstance(d[key]["data"],torch.Tensor):
                    d[key]["data"] = d[key]["data"].to(self.device)

        d = self.recursive_apply(d)
        return d

class WeightedStitcher(StitcherBase):
    def __init__(self, edge_to_center_ratio, **kwargs):
        super().__init__(**kwargs)
        self.edge_to_center_ratio = edge_to_center_ratio
        assert edge_to_center_ratio > 0 and edge_to_center_ratio < 1
        self.weight = self.get_weight().unsqueeze(0).unsqueeze(0)

    def forward(self,model,patch_loader,grid_sampler):
        aggregator = tio.inference.GridAggregator(grid_sampler,overlap_mode="average")
        aggregator_weight = tio.inference.GridAggregator(grid_sampler,overlap_mode="average")
        self.weight = self.weight.to(self.device)
        for patches_batch in tqdm(patch_loader,desc="predicting patches"):
            patches_batch = self.move_dict_to_device(patches_batch)
            pred = model(patches_batch["ct"]["data"])
            n_batch = pred.shape[0]
            w = self.weight.expand(n_batch,-1,-1,-1,-1)
            locations = patches_batch[tio.LOCATION]
            aggregator.add_batch(pred*w, locations)
            aggregator_weight.add_batch(w, locations)

        img = aggregator.get_output_tensor()
        w = aggregator_weight.get_output_tensor()
        self.weight = self.weight.cpu()
        return img/w
    
    @abstractmethod
    def get_weight(self):
        pass


class GennormStitcher(WeightedStitcher):
    def __init__(self,edge_to_center_ratio, beta,**kwargs):
        self.beta = beta
        super().__init__(edge_to_center_ratio,**kwargs)

    def _get_scale(self):
        p = self.edge_to_center_ratio
        scale = (np.array(self.patch_size)/2)/((-np.log(p))**(1/self.beta))
        return scale
    
    def get_weight(self):
        scales = self._get_scale()
        g1 = scipy.stats.gennorm.pdf(np.arange(self.patch_size[0]),  beta=self.beta, loc=self.patch_size[0]/2, scale=scales[0])
        g2 = scipy.stats.gennorm.pdf(np.arange(self.patch_size[1]),  beta=self.beta, loc=self.patch_size[1]/2, scale=scales[1])
        g3 = scipy.stats.gennorm.pdf(np.arange(self.patch_size[2]),  beta=self.beta, loc=self.patch_size[2]/2, scale=scales[2])
        g1 = g1/g1.max()
        g2 = g2/g2.max()
        g3 = g3/g3.max()
        kernel = g1[:,np.newaxis,np.newaxis]*g2[np.newaxis,:,np.newaxis]*g3[np.newaxis,np.newaxis,:]
        return torch.from_numpy(kernel)

