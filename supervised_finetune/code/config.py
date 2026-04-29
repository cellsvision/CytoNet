from easydict import EasyDict


train_cfg = EasyDict()

train_cfg.GPU_ID = "0, 1, 2, 3"
train_cfg.train_csv_path = '../datalists/train.csv'
train_cfg.val_csv_path = '../datalists/val.csv'
train_cfg.patch_root = '../sample_data'
train_cfg.checkpoint_path = '../checkpoint/teacher_backbone_torchvision.pth'
train_cfg.train_patch_class_id = {'nilm':0,'auc':1,'hguc':2, 'impurity':3, 'histiocyte':4, 'blank':5}
train_cfg.wsi_class_list = ['blank', 'histiocyte', 'impurity', 'yin', 'AUC', 'HGUC'] 
train_cfg.train_patch_class = {'nilm': ['NHGUC', 'NILM'], 'auc': ['AUC', 'LGUN'], 'hguc': ['HGUC', 'AUC_H', 'CA', 'SHGUC'], 'impurity': ['IMPURITY'], 'histiocyte': ['HISTIOCYTE'], 'blank': ['BLANK']}
train_cfg.use_n_others = 12
train_cfg.use_his = 4
train_cfg.use_imp = 4
train_cfg.use_blk = 4
train_cfg.target_size = 1024

train_cfg.enhance = 'AUC'
train_cfg.enhance_times = 0

train_cfg.preprocessing_func = lambda x:x
train_cfg.batch_size = 4 # 16
train_cfg.max_epoch = 30

train_cfg.weight_path = None

train_cfg.save_model_dir = '../models'

train_cfg.do_train = True
train_cfg.eval_ckpt = ''
