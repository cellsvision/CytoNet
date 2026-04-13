import os
import random

from PIL import Image
import numpy as np
import cv2

import torch.utils.data as data

def scan_files(input_file_path, ext_list = ['.txt'], replace_root=True):
    file_list = []
    for root, dirs, files in os.walk(input_file_path):
        # scan all files and put it into list

        for f in files:
            if os.path.splitext(f)[1].lower() in ext_list:
                if replace_root:
                    file_list.append(os.path.join(root, f).replace("\\","/").replace(os.path.join(input_file_path, "").replace("\\","/"), "", 1 ))
                else:
                    file_list.append(os.path.join(root, f).replace("\\","/"))

    return file_list

class CSVDataset(data.Dataset):
    def __init__(self, source_list, 
                 file_length=None,
                 img_ext=['.png', '.jpeg', '.jpg', '.tif', '.tiff'],
                 target_image_size=[256, 3072],
                 transform=None):

        super(CSVDataset, self).__init__()

        self._source_list = source_list

        self._file_names_list = self._get_file_names(ext_list=img_ext)
        self._file_length = file_length

        self._target_img_size = target_image_size

        self.transform = transform

        self._dataset_index_list = np.random.randint(len(self._source_list), size=self.__len__())

    def __len__(self):
        if self._file_length is not None:
            return self._file_length

        total_files_conut = sum([len(f_list) for f_list in self._file_names_list])

        
        # return len(self._file_names)
        return total_files_conut


    def __getitem__(self, index):


        '''
        select dataset
        '''
        dataset_index = self._dataset_index_list[index]
        
        current_file_name_list = self._file_names_list[dataset_index]

        
        true_idx = random.randint(0, len(current_file_name_list)-1)

        img_path = os.path.join(self._source_list[dataset_index], 
                                        current_file_name_list[true_idx])
        
        '''
        read image and gt 
        gt file: a mask with shape H*W, the pixel value is the class index
        '''
        whole_img_array = self._fetch_data(img_path)

        h,w = whole_img_array.shape[:2]

        current_target_size = min(h,w)

        current_target_img_size = random.randint(min(self._target_img_size[0], current_target_size), min(self._target_img_size[1], current_target_size))
        y_range = h - current_target_img_size
        x_range = w - current_target_img_size

        start_x = random.randint(0, max(0,x_range-1))
        start_y = random.randint(0, max(0,y_range-1))
        end_x = start_x + current_target_img_size
        end_y = start_y + current_target_img_size

        img = np.array(whole_img_array[start_y:end_y, start_x:end_x])
        
        # ori_img = np.array(img)

        if self.transform is not None:
            final_img = Image.fromarray(img)
            final_img = self.transform(final_img)

        return (final_img, 1)


    def _get_file_names(self, ext_list = ['.png']):


        file_names_list = []

        for data_dir in self._source_list:
            # tmp_bg_file_list = []
            # tmp_fg_file_list = []
            # img_path = os.path.join(data_dir, self._image_dir)
            # print("#### debug get file names")
            # print(img_path)
            # print("#### debug get file name done")
            files_list = scan_files(data_dir, ext_list=ext_list)
            # mask_path = os.path.join(data_dir, self._label_dir)            
            
            file_names_list.append(files_list)
        
        return file_names_list


    def _fetch_data(self, img_path):
        img = self._open_image(img_path)

        return img

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        # cv2: B G R
        # h w c
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        img = img[:,:,:3]
        img = img[:,:,::-1]

        return img

class CSVDataset_v2(data.Dataset):
    def __init__(self, source_list, 
                 file_length=None,
                 img_ext=['.png', '.jpeg', '.jpg', '.tif', '.tiff'],
                 source_ratio=None,
                 transform=None):

        super(CSVDataset, self).__init__()

        self._source_list = source_list

        self._file_names_list = self._get_file_names(ext_list=img_ext)
        self._file_length = file_length
        self._source_ratio = source_ratio
        # self._target_img_size = target_image_size

        self.transform = transform

        # self._dataset_index_list = np.random.randint(len(self._source_list), size=self.__len__())
        self._dataset_index_list = random.choices(list(len(self._source_list)), self._source_ratio, k=self.__len__())

    def __len__(self):
        if self._file_length is not None:
            return self._file_length

        total_files_conut = sum([len(f_list) for f_list in self._file_names_list])

        
        # return len(self._file_names)
        return total_files_conut


    def __getitem__(self, index):


        '''
        select dataset
        '''
        dataset_index = self._dataset_index_list[index]
        
        current_file_name_list = self._file_names_list[dataset_index]

        
        true_idx = random.randint(0, len(current_file_name_list)-1)

        img_path = os.path.join(self._source_list[dataset_index], 
                                        current_file_name_list[true_idx])
        
        '''
        read image and gt 
        gt file: a mask with shape H*W, the pixel value is the class index
        '''
        # whole_img_array = self._fetch_data(img_path)
        img = self._fetch_data(img_path)

        # h,w = whole_img_array.shape[:2]

        # current_target_size = min(h,w)

        # current_target_img_size = random.randint(min(self._target_img_size[0], current_target_size), min(self._target_img_size[1], current_target_size))
        # y_range = h - current_target_img_size
        # x_range = w - current_target_img_size

        # start_x = random.randint(0, max(0,x_range-1))
        # start_y = random.randint(0, max(0,y_range-1))
        # end_x = start_x + current_target_img_size
        # end_y = start_y + current_target_img_size

        # img = np.array(whole_img_array[start_y:end_y, start_x:end_x])
        
        # ori_img = np.array(img)

        if self.transform is not None:
            final_img = Image.fromarray(img)
            final_img = self.transform(final_img)

        return (final_img, 1)


    def _get_file_names(self, ext_list = ['.png']):


        file_names_list = []

        for data_dir in self._source_list:
            # tmp_bg_file_list = []
            # tmp_fg_file_list = []
            # img_path = os.path.join(data_dir, self._image_dir)
            # print("#### debug get file names")
            # print(img_path)
            # print("#### debug get file name done")
            files_list = scan_files(data_dir, ext_list=ext_list)
            # mask_path = os.path.join(data_dir, self._label_dir)            
            
            file_names_list.append(files_list)
        
        return file_names_list


    def _fetch_data(self, img_path):
        img = self._open_image(img_path)

        return img

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        # cv2: B G R
        # h w c
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        img = img[:,:,:3]
        img = img[:,:,::-1]

        return img
    
import pickle
class CSVDataset_v3(data.Dataset):
    def __init__(self, path_pickle, 
                 file_length=None,
                 img_ext=['.png', '.jpeg', '.jpg', '.tif', '.tiff'],
                 transform=None):

        super(CSVDataset_v3, self).__init__()

        # self._source_dir = source_dir
        with open(path_pickle, 'rb') as f:
            self._file_names_list = pickle.load(f)

        print('### debug file lenght: ', len(self._file_names_list))
        # self._file_names_list = scan_files(source_dir, ext_list=img_ext)
        self._file_length = file_length
        # self._target_img_size = target_image_size

        self.transform = transform
        random.shuffle(self._file_names_list)

        # self._dataset_index_list = np.random.randint(len(self._source_list), size=self.__len__())
        # self._dataset_index_list = random.choices(list(len(self._source_list)), self._source_ratio, k=self.__len__())

    def __len__(self):
        if self._file_length is not None:
            return self._file_length

        # total_files_conut = sum([len(f_list) for f_list in self._file_names_list])
        total_files_conut = len(self._file_names_list)

        
        # return len(self._file_names)
        return total_files_conut


    def __getitem__(self, index):


        '''
        select dataset
        '''
        # dataset_index = self._dataset_index_list[index]
        
        # current_file_name = self._file_names_list[index]
        img_path = self._file_names_list[index]

        

        # img_path = os.path.join(self._source_dir, current_file_name)
        
        '''
        read image and gt 
        gt file: a mask with shape H*W, the pixel value is the class index
        '''
        # whole_img_array = self._fetch_data(img_path)
        img = self._fetch_data(img_path)

        # h,w = whole_img_array.shape[:2]

        # current_target_size = min(h,w)

        # current_target_img_size = random.randint(min(self._target_img_size[0], current_target_size), min(self._target_img_size[1], current_target_size))
        # y_range = h - current_target_img_size
        # x_range = w - current_target_img_size

        # start_x = random.randint(0, max(0,x_range-1))
        # start_y = random.randint(0, max(0,y_range-1))
        # end_x = start_x + current_target_img_size
        # end_y = start_y + current_target_img_size

        # img = np.array(whole_img_array[start_y:end_y, start_x:end_x])
        
        # ori_img = np.array(img)

        if self.transform is not None:
            final_img = Image.fromarray(img)
            final_img = self.transform(final_img)

        return (final_img, 1)


    # def _get_file_names(self, ext_list = ['.png']):


    #     file_names_list = []

    #     for data_dir in self._source_list:
    #         # tmp_bg_file_list = []
    #         # tmp_fg_file_list = []
    #         # img_path = os.path.join(data_dir, self._image_dir)
    #         # print("#### debug get file names")
    #         # print(img_path)
    #         # print("#### debug get file name done")
    #         files_list = scan_files(data_dir, ext_list=ext_list)
    #         # mask_path = os.path.join(data_dir, self._label_dir)            
            
    #         file_names_list.append(files_list)
        
    #     return file_names_list


    def _fetch_data(self, img_path):
        img = self._open_image(img_path)

        return img

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        # cv2: B G R
        # h w c
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        img = img[:,:,:3]
        img = img[:,:,::-1]

        return img