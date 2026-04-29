"""
Urine cell classification training and evaluation script.
Uses RegNet-Y with CBAM attention for multi-class classification.
"""

import pandas as pd
import numpy as np
from glob import glob
from sklearn.utils import shuffle
import albumentations as A
import logging
from tqdm import tqdm
import time
from PIL import Image
import os, sys
from scipy.special import softmax
import cv2
import random
from sklearn import metrics
from copy import deepcopy

import timm
from timm.models.layers.cbam import CbamModule

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.models import regnet_y_800mf

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from config import train_cfg

# Class labels for classification
all_cls = ['NILM', 'AUC', 'HGUC', 'IMPURITY', 'HISTIOCYTE', 'BLANK']
eval_interval = 1

os.environ["CUDA_VISIBLE_DEVICES"] = train_cfg.GPU_ID
device = torch.device("cuda")

logging.basicConfig(filename='../logs/log_sft_train.log', level=logging.INFO)


def get_file_path(df):
    """Extract image file paths and labels from WSI directories."""
    wsi_dir = np.array(df['wsi_dir'])
    wsi_slide = []
    image_pathes = []
    patch_label = []
    patch_cls = []
    
    for wsi in wsi_dir:
        # Process patch classes from config
        for curr_cls, curr_labels in train_cfg.train_patch_class.items():
            for curr_label in curr_labels:
                files = list(glob(wsi + '/*/*_' + curr_label + '.png'))
                if len(files) > 0:
                    wsi_slide.extend([wsi] * len(files))
                    image_pathes.extend(files)
                    patch_cls.extend([curr_cls] * len(files))
                    patch_label.extend([curr_label] * len(files))

        # Process NILM patches
        if len(list(glob(wsi + '/NILM/*)_NILM.png'))) > 0:
            files = list(glob(wsi + '/NILM/*)_NILM.png'))
            wsi_slide.extend([wsi] * len(files))
            image_pathes.extend(files)
            patch_cls.extend(['nilm'] * len(files))
            patch_label.extend(['NILM'] * len(files))
        if len(list(glob(wsi + '/NILM/*.png'))) > 0:
            files = list(glob(wsi + '/NILM/*.png'))
            wsi_slide.extend([wsi] * len(files))
            image_pathes.extend(files)
            patch_cls.extend(['nilm'] * len(files))
            patch_label.extend(['NILM'] * len(files))
        
        # Process IMPURITY patches
        if len(list(glob(wsi + '/IMPURITY/*.png'))) > 0:
            files = list(glob(wsi + '/IMPURITY/*.png'))
            wsi_slide.extend([wsi] * len(files))
            image_pathes.extend(files)
            patch_cls.extend(['impurity'] * len(files))
            patch_label.extend(['IMPURITY'] * len(files))
        
        # Process HISTIOCYTE patches
        if len(list(glob(wsi + '/HISTIOCYTE/*.png'))) > 0:
            files = list(glob(wsi + '/HISTIOCYTE/*.png'))
            wsi_slide.extend([wsi] * len(files))
            image_pathes.extend(files)
            patch_cls.extend(['histiocyte'] * len(files))
            patch_label.extend(['HISTIOCYTE'] * len(files))

    new_df = pd.DataFrame()
    new_df['slide'] = wsi_slide
    new_df['image_pathes'] = image_pathes
    new_df['patch_label'] = patch_label
    new_df['patch_cls'] = patch_cls
    new_df = shuffle(new_df)
    print('---------', len(new_df))
    return new_df


def get_wsi_path(root_path, class_list):
    """Get WSI paths for specified classes."""
    WSIclass = []
    WSIdir = []
    for i, c in enumerate(class_list):
        wsis = list(glob(root_path + '/' + c + '/*'))
        WSIdir.extend(wsis)
        WSIclass.extend([c] * len(wsis))
    df = pd.DataFrame()
    df['wsi_dir'] = WSIdir
    df['wsi_cls'] = WSIclass
    return df


def get_file_path_from_csv(csv_path):
    """Load file paths from CSV with optional class enhancement."""
    fns_df = pd.read_csv(csv_path, dtype=str)['filename'].values
    WSI_df = get_wsi_path(train_cfg.patch_root, train_cfg.wsi_class_list)
    
    # Apply class enhancement if configured
    if train_cfg.enhance is not None:
        WSI_df_tmp = [WSI_df[WSI_df['wsi_cls'] == train_cfg.enhance]] * train_cfg.enhance_times
        WSI_df = pd.concat([WSI_df] + WSI_df_tmp)
    
    WSI_df = WSI_df[WSI_df['wsi_dir'].apply(lambda x: os.path.basename(x)).isin(fns_df)]
    file_path_df = get_file_path(WSI_df)
    return file_path_df


class UrineDataset(Dataset):
    """Custom dataset for urine cell image classification."""
    
    def __init__(self, data_df, section='train', preprocessing_func=None):
        self.section = section
        self.data_df = data_df
        self.label_list = train_cfg.train_patch_class_id
        self.aug_seq = self.create_aug_seq()
        self.preprocessing_func = preprocessing_func
        self.on_epoch_start()

    def on_epoch_start(self):
        """Sample and balance dataset at each epoch start."""
        if self.section == 'train' or self.section == 'val':
            tmp_filepath = []
            tmp_cls = []
            df = shuffle(self.data_df)
            
            # Split by patch label
            histioctye = df[(df['patch_label'] == 'HISTIOCYTE')].copy()
            impurity = df[(df['patch_label'] == 'IMPURITY')].copy()
            blank = df[(df['patch_label'] == 'BLANK')].copy()
            front = df[~((df['patch_label'] == 'HISTIOCYTE') | (df['patch_label'] == 'IMPURITY') | (df['patch_label'] == 'BLANK'))].copy()
            neg = front[(front['patch_label'] == 'NILM') | (front['patch_label'] == 'NHGUC')].copy()
            pos = front[(front['patch_label'] == 'AUC_H') | (front['patch_label'] == 'HGUC') | (front['patch_label'] == 'SHGUC')].copy()
            auc = front[(front['patch_label'] == 'AUC')]
            
            # Sample per slide to balance dataset
            histioctye_filter = histioctye.groupby(['slide']).head(train_cfg.use_his)
            impurity_filter = impurity.groupby(['slide']).head(train_cfg.use_imp)
            blank_filter = blank.groupby(['slide']).head(train_cfg.use_blk)
            neg_filter = neg.groupby(['slide']).head(train_cfg.use_n_others)
            
            new_df = pd.concat([pos, auc, histioctye_filter, neg_filter, impurity_filter, blank_filter])
            new_df = shuffle(new_df)
            new_df = shuffle(new_df)
            new_df = shuffle(new_df)
            print('num of whole pos', len(pos), len(auc))
            print('num of whole neg', 'neg:', len(neg_filter), 'his:', len(histioctye_filter), 'imp:', len(impurity_filter), 'blk:', len(blank_filter))
            print('num of whole data', len(new_df))
            
            for i, row in new_df.iterrows():
                if row['patch_label'] in ['bg_others']:
                    tmp_f = glob(row['slide'] + '/' + row['patch_label'] + '/*.png')
                    tmp_f = np.random.choice(tmp_f, min(train_cfg.use_n_others, len(tmp_f)), replace=False)
                    tmp_filepath.extend(tmp_f)
                    tmp_cls.extend([row['patch_cls']] * len(tmp_f))
                else:
                    tmp_filepath.extend([row['image_pathes']] * 1)
                    tmp_cls.extend([row['patch_cls']] * 1)
        else:
            raise NotImplementedError

        self.curr_df = pd.DataFrame()
        self.curr_df['filepath'] = tmp_filepath
        self.curr_df['cls'] = tmp_cls
        self.curr_df = shuffle(self.curr_df)
        print('---------', len(self.curr_df))
        print(self.curr_df)

    def __len__(self):
        return len(self.curr_df)

    def __getitem__(self, indx):
        filepath = self.curr_df.loc[indx, 'filepath']
        label_cls = self.curr_df.loc[indx, 'cls']
        # Create soft label with smoothing
        label = np.zeros((len(self.label_list),), dtype=np.uint8) + (0.05 / (len(self.label_list) - 1))
        label[self.label_list[label_cls]] = 0.95
        im, label = self.get_image(filepath, label)
        im = np.transpose(im, [2, 0, 1])
        return im, label

    def get_empty_image(self, label, image_path):
        """Return empty image for invalid/corrupted images."""
        im = np.zeros((train_cfg.target_size, train_cfg.target_size, 3), dtype=np.uint8)
        label[0] = 1
        label[1:] = 0
        return im, label

    def get_image(self, image_path, label):
        """Load and preprocess image from path."""
        if not os.path.exists(image_path):
            return self.get_empty_image(label, image_path)
        
        im = np.array(Image.open(image_path))
        if len(im.shape) != 3 or im.shape[-1] != 3:
            return self.get_empty_image(label, image_path)

        if im.shape[1] < train_cfg.target_size or im.shape[0] < train_cfg.target_size or np.min(im) < 0 or np.max(im) > 255:
            return self.get_empty_image(label, image_path)
        h, w, _ = im.shape
        
        # Random crop if image is larger than target size
        if (h > train_cfg.target_size) or (w > train_cfg.target_size):
            im_list = []
            choice = [1]
            for _ in range(np.random.choice(choice, 1)[0]):
                x = random.randint(0, im.shape[1] - train_cfg.target_size)
                y = random.randint(0, im.shape[0] - train_cfg.target_size)
                im_tmp = im[y:y + train_cfg.target_size, x:x + train_cfg.target_size]
                im_list.append(im_tmp)
            im_list = np.array(im_list)
            if np.random.random() < 0.5:
                im = np.mean(im_list, axis=0).astype(np.uint8)
            else:
                im = np.min(im_list, axis=0).astype(np.uint8)

        # Apply augmentation for training
        try:
            if self.section == 'train':
                im = self.aug_seq(image=im)['image']
        except Exception as e:
            print('======', e, type(self.aug_seq(image=im)))
            print('========', image_path)

        assert im.shape == (train_cfg.target_size, train_cfg.target_size, 3), str(im.shape)
        im = self.preprocessing_func(np.expand_dims(im, 0))[0]
        return im, label

    def create_aug_seq(self):
        """Create augmentation pipeline using albumentations."""
        seq = A.Compose([
            A.JpegCompression(p=0.3, quality_lower=70, quality_upper=95),
            A.OneOf([
                A.RandomGamma(gamma_limit=[50, 150], p=0.5),
                A.CLAHE(clip_limit=2.5, tile_grid_size=(80, 80), p=0.5),
                A.RandomBrightnessContrast(p=1.0, brightness_limit=[-0.4, 0.4], contrast_limit=[-0.25, 0.21]),
            ]),
            A.Flip(p=0.7),
            A.OneOf([
                A.HueSaturationValue(p=0.5, hue_shift_limit=[-10, 10], sat_shift_limit=[-45, 25], val_shift_limit=[0.0, 0.0]),
                A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, always_apply=False, p=0.4),
            ]),
            A.OneOf([
                A.ISONoise(p=0.5, color_shift=(0.01, 0.2), intensity=(0.08, 0.13)),
                A.MultiplicativeNoise(p=0.5, multiplier=(0.65, 1.2), per_channel=False),
                A.GaussNoise(p=0.5, var_limit=(10, 150)),
            ])
        ])
        return seq


class UrineModel(nn.Module):
    """RegNet-Y with CBAM attention for urine cell classification."""
    
    def __init__(self):
        super(UrineModel, self).__init__()
        regnet = regnet_y_800mf(pretrained=False)
        
        # Handle different torchvision versions
        if hasattr(regnet, 'features'):
            self.model_fea = regnet.features
        else:
            self.model_fea = nn.Sequential(regnet.stem, regnet.trunk_output)
        
        self.cbam = CbamModule(channels=784, rd_ratio=1/16)
        self.model_cls_map = nn.Conv2d(in_channels=784, out_channels=6, kernel_size=3, stride=1)
        self.model_out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.model_fea(x)
        x = self.cbam(x)
        cam_out = self.model_cls_map(x)
        final_out = self.model_out(cam_out)
        return final_out


def load_model():
    """Load model with optional pretrained weights."""
    print('loading model')
    model = UrineModel()
    
    if train_cfg.checkpoint_path is not None:
        print('loading pretrained model {}'.format(train_cfg.checkpoint_path))
        stdict = torch.load(train_cfg.checkpoint_path)
        new_stdict = {}
        model_keys = set(model.state_dict().keys())
        
        # Map pretrained weight keys to model keys
        for k, v in stdict.items():
            if k.startswith('stem.'):
                new_key = 'model_fea.0.' + k[5:]
                if new_key in model_keys:
                    new_stdict[new_key] = v
            elif k.startswith('trunk_output.'):
                new_key = 'model_fea.1.' + k[13:]
                if new_key in model_keys:
                    new_stdict[new_key] = v
        
        missing_keys = [i for i in model.state_dict().keys() if i not in new_stdict.keys()]
        print('missing_keys', missing_keys)
        print('loaded keys count:', len(new_stdict))
        model.load_state_dict(new_stdict, strict=False)
    
    model = model.to(device)
    model = nn.DataParallel(model)
    return model


def load_trained_model(device, cpu_only=False):
    """Load trained model checkpoint for evaluation."""
    print('loading model')
    model = UrineModel()
    print('loading ckpt model {}'.format(train_cfg.eval_ckpt))
    
    if cpu_only:
        stdict = torch.load(train_cfg.eval_ckpt, map_location=torch.device('cpu'))
    else:
        stdict = torch.load(train_cfg.eval_ckpt)
    
    # Remove 'module.' prefix from DataParallel saved weights
    new_stdict = {}
    for i, (k, v) in enumerate(stdict.items()):
        if 'module.' in k:
            new_stdict[k[7:]] = stdict[k]
        else:
            new_stdict[k] = stdict[k]
    model.load_state_dict(new_stdict, strict=True)
    
    model = model.to(device)
    model = nn.DataParallel(model)
    print(model)
    return model


def evaluate_model(model, test_data_loader):
    """Evaluate model and compute accuracy metrics."""
    model.eval()
    pred_all = []
    label_all = []
    
    with torch.no_grad():
        for batch_id, batch_data in enumerate(test_data_loader):
            images, labels = batch_data
            images = images.float().to(device)
            labels = np.argmax(labels, axis=1)
            label_all.extend(labels.numpy())
            labels = labels.float().to(device)
            pred_labels = model(images)
            pred_all.extend(pred_labels.cpu().detach().numpy())
    
    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    m = softmax(np.array(pred_all), axis=1)
    pred_label = np.argmax(pred_all, axis=1)
    acc = np.mean(pred_label == label_all)
    
    # Compute binary accuracy (positive vs negative)
    pred_label_2 = deepcopy(pred_label)
    pred_label_2[pred_label_2 == 3] = 0
    pred_label_2[pred_label_2 == 4] = 0
    pred_label_2[pred_label_2 == 5] = 0
    pred_label_2[pred_label_2 == 1] = 2
    
    label_all_2 = deepcopy(label_all)
    label_all_2[label_all_2 == 3] = 0
    label_all_2[label_all_2 == 4] = 0
    label_all_2[label_all_2 == 5] = 0
    label_all_2[label_all_2 == 1] = 2
    acc_2 = np.mean(pred_label_2 == label_all_2)
    
    return acc, acc_2, label_all, pred_label, label_all_2, pred_label_2


if __name__ == "__main__":
    # Prepare training data
    if train_cfg.do_train:
        train_df = get_file_path_from_csv(train_cfg.train_csv_path)
        print(train_df)
        data_train = UrineDataset(train_df, section='train', preprocessing_func=train_cfg.preprocessing_func)
        train_data_loader = DataLoader(data_train, batch_size=train_cfg.batch_size, shuffle=True, pin_memory=False, num_workers=16)
    
    # Prepare test data
    test_df = get_file_path_from_csv(train_cfg.val_csv_path)
    data_test = UrineDataset(test_df, section='val', preprocessing_func=train_cfg.preprocessing_func)
    test_data_loader = DataLoader(data_test, batch_size=train_cfg.batch_size, shuffle=True, pin_memory=False, num_workers=12)

    if train_cfg.do_train:
        # Training setup
        model = load_model()
        optimizer = torch.optim.Adam(model.parameters(), 5e-6)
        schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-7)
        loss_function = torch.nn.CrossEntropyLoss()
        batches_per_epoch = len(train_data_loader)
        writer = SummaryWriter(log_dir=train_cfg.save_model_dir + '/logs', flush_secs=10)

        # Training loop
        for epoch in range(train_cfg.max_epoch):
            logging.info('Start epoch {}'.format(epoch))
            model.train()
            epoch_loss = 0
            train_time_sp = time.time()
            data_train.on_epoch_start()
            
            for batch_id, batch_data in enumerate(train_data_loader):
                images, labels = batch_data
                images = images.float().to(device)
                labels = np.argmax(labels, axis=1)
                labels = labels.long().to(device)
                optimizer.zero_grad()
                pred_labels = model(images)

                loss_value = loss_function(pred_labels, labels)
                loss_value.backward()
                optimizer.step()

                epoch_loss += loss_value.item()
                used_time = (time.time() - train_time_sp) / (batch_id + 1)
                writer.add_scalar('Loss/train', epoch_loss / (1 + batch_id), batches_per_epoch * epoch + batch_id)
                logging.info(f'epoch {epoch} Batch: {batch_id}, loss = {loss_value.item():.3f}, avg_batch_time = {used_time:.3f}')
                print('====')
            
            # Evaluation at intervals
            if epoch % eval_interval == 0:
                acc, acc_2, label_all, pred_label, label_all_2, pred_label_2 = evaluate_model(model, test_data_loader)
                logging.info(f'=========================== acc {acc} acc_2 {acc_2} =================')
                writer.add_scalar('ACC/acc_4', acc, epoch)
                writer.add_scalar('ACC/acc_2', acc_2, epoch)
                torch.save(model.state_dict(), train_cfg.save_model_dir + f'/ckpt_{epoch}_{acc}_{acc_2}.pth')
    else:
        # Evaluation only mode
        model = load_trained_model(device=device)
        acc, acc_2, label_all, pred_label, label_all_2, pred_label_2 = evaluate_model(model, test_data_loader)
        logging.info(f'=========================== acc {acc} acc_2 {acc_2} =================')
        print(metrics.confusion_matrix(label_all, pred_label, labels=[0, 1, 2, 3, 4, 5]))
        print(metrics.confusion_matrix(label_all_2, pred_label_2, labels=[0, 2]))
        print(metrics.classification_report(label_all, pred_label, digits=6, target_names=all_cls))