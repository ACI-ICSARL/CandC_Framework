import torch
from torch.utils.data import Dataset,DataLoader

class RegressionData(Dataset):
    """Simple Regression Dataset"""

    def __init__(self, X, Y, transform=None,device='cpu'):
        """
        Args:
            path (string): Path to the Acoustic folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.x = X.detach().clone().cpu()#.to(device)
        self.y = Y.detach().clone().cpu()#.to(device)
        self.transform = transform
        self.device = device
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            inputs = self.x[idx,:]
            output = self.y[idx]
            inputs= inputs.to(self.device)
            output= output.to(self.device)
            sample = (inputs,output)#(self.x[idx].to(self.device), self.y[idx].to(self.device))

            if self.transform:
                sample = self.transform(sample)
        except IndexError as ie:
            print("Raised the following index error: {}".format(ie))
            print("The first 'x' datum is: {}".format(self.x[0]))
            print("'x' has shape: {}".format(self.x.shape))
            sample = (torch.Tensor([]),torch.Tensor([]))
        return sample

def convert_to_reg_dataset(X:torch.Tensor,Y:torch.Tensor,transform=None,device='cpu'):
        """ Function to Convert X and Y inputs into an instance of the RegressionData class
        
        Parameters
        --------------------
        :X: torch.Tensor, input features
        :Y: torch.Tensor, output labels/features
        :transform: optional preprocessing transform to apply to X and y
        :device: indicates what cuda device or cpu processing should take place on
        """
        dataset= RegressionData(X,Y,transform=transform,device=device)
        return dataset

def convert_to_reg_dataloader(reg_dataset,batch_size=64,shuffle=False,num_workers=64,pin_memory=False):
    """ Function to Convert a RegressionData class into a DataLoader
    """
    return DataLoader(reg_dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                     pin_memory=pin_memory)

def make_reg_dataloader(X,Y,transform=None,device='cpu',batch_size=64, shuffle=False, num_workers=64,pin_memory=False):
    """ Composition of convert_to_reg_dataset with convert_to_reg_dataloader, to produce an DataLoader given X,Y and parameters"""
    return convert_to_reg_dataloader(convert_to_reg_dataset(X,Y,transform=transform,device=device),
                                    batch_size=batch_size,
                                    shuffle=shuffle, 
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)
