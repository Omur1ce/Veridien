import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class ChungusDataset(Dataset):
    def __init__(self, df, transform=None):
        # read the shi
        #self.img_paths = df['img_path'].tolist()
        self.lidar_paths = df["lidar_file"].tolist()
        self.mask_paths = df['mask_file'].tolist()
        self.topleft = df['topleft'].tolist()
        self.width = df['width'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.lidar_paths)


    def __getitem__(self, idx):
        #img = SOME_READING_FUNCTION(self.img_paths[idx]) # READ IN (C, H , W) please
        lidar = SOME_READING_FUNCTION(self.lidar_paths[idx]) # READ IN (1, H, W)
        mask = SOME_READING_FUNCTION(self.mask_paths[idx]) # just (H, W) is fine, but (1, H, W) is fine too

        topleft = self.topleft[idx] # (r, c)
        width = self.width[idx]

        # norm img, liadr to 0, 1
        #img = img.astype(np.float32) / 255.0  
        lidar = (lidar - np.min(lidar)) / (np.max(lidar) - np.min(lidar))
        mask = mask.squeeze().astype(np.int64) 
        
        # crop img, lidar, mask
        #img = img[:, topleft[0]:topleft[0]+width, topleft[1]:topleft[1]+width]  # (3, width, width)
        lidar = lidar[:, topleft[0]:topleft[0]+width, topleft[1]:topleft[1]+width] # (1, width, width)
        mask = mask[topleft[0]:topleft[0]+width, topleft[1]:topleft[1]+width]
        
        #combined_img = np.vstack((img, lidar))
        combined_img = lidar
        
        # augmentation
        if self.transform:
            augmented = self.transform(image=combined_img.transpose(1, 2, 0), mask=mask) # bruh its need to be h, w, c
            final_img = augmented['image'].transpose(2, 0, 1) # ok its back to c, h, w
            final_mask = augmented['mask']

        return torch.tensor(final_img, dtype=torch.float32), torch.tensor(final_mask, dtype=torch.long)


# train transform with alumbentniatnotnos
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(45),
    # A.RandomCrop(224, 224),
    A.Resize(height=224, width=224), # maybe this is better idk test it https://explore.albumentations.ai/ more augmentations
    #ToTensorV2(),
])


val_test_transforms = A.Compose([
   A.Resize(height=224, width=224),
    ToTensorV2(),
])


def split_dataset(input_df: pd.DataFrame, splits=(0.8, 0.1, 0.1), random_state=69):
    """
    Splits df to train val and test
     - `input_df` original df
     - `splits` tuple of (train, val, test) splits, sum to 1
     - `random_state` rng seed (69 haha)
     returns test_df, val_df, test_df from df.
    """
    assert sum(splits) <= 1.0, "Train/Val/Test splits sum over 1."

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    train_df, temp_df = train_test_split(input_df, test_size=(1-splits[0]), random_state=random_state, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=(splits[2]/(splits[1]+splits[2])), random_state=random_state, shuffle=True)

    print(f"Split dataset: {train_df.shape=}, {val_df.shape=}, {test_df.shape=}")
    return train_df, val_df, test_df


def create_dataloader(
    input_df: pd.DataFrame,
    splits=(0.8, 0.1, 0.1),
    batch_size=32, random_state=69,
    n_workers=4, pin_mem=True,
    train_transform=train_transforms,
    val_test_transform=val_test_transforms,
    goofy_multiplier = 1,
):
    """
    Splits `input_df` then creates and returns dataloaders.
     - `splits` tuple of (train, val, test) splits, sum to 1
     - `batch_size` default 32, turn it up in powers of 2 of u got enough vram sob
     - `random_state` rng seed
     - `n_workers` dataloader workers turn that shit up if u have more cpu cores (to like half of them maybe idk)
     - `pin_mem` yep
     - `train_transform`, `test_transform` augmentations
     - `goofy_multiplier` how many times to repeat the dataset (this might not work well idk)
     returns train, val, test dataloaders
    """
    train_df, val_df, test_df = split_dataset(input_df, splits, random_state)
    
    
    train_df_large = pd.concat([train_df]*goofy_multiplier, ignore_index=True).sample(frac=1).reset_index(drop=True)

    train_dataset = ChungusDataset(train_df_large, transform=train_transform)
    val_dataset = ChungusDataset(val_df, transform=val_test_transform)
    test_dataset = ChungusDataset(test_df, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    return train_loader, val_loader, test_loader

