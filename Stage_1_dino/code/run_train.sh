

export CUDA_VISIBLE_DEVICES=0,1,2,3

# # cells regnet
# torchrun --nproc_per_node=4 main_dino.py --arch regnet_y_800mf \
#         --valid_data_pkl ../datalists/sample_cells_valid_img_path.pkl \
#         --output_dir ../models \
#         --batch_size_per_gpu 4 \
#         --saveckp_freq 1

python3 -m torch.distributed.run --nproc_per_node=4 main_dino.py --arch regnet_y_800mf \
        --valid_data_pkl ../datalists/sample_cells_valid_img_path.pkl \
        --output_dir ../models \
        --batch_size_per_gpu 4 \
        --saveckp_freq 1