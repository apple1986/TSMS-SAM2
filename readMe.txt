# train
CUDA_VISIBLE_DEVICES=0 python training/train.py --config configs/sam2.1_training/sam2.1_hiera_s_endovis18_instrument

# test
python tools/vos_inference.py --sam2_cfg configs/sam2.1/sam2.1_hiera_s.yaml --sam2_checkpoint ./checkpoints/sam2.1_hiera_s_endo18.pth --output_mask_dir ./results/sam2.1/endovis_2018/instrument --input_mask_dir ./datasets/VOS-Endovis18/valid/VOS/Annotations_vos_instrument --base_video_dir ./datasets/VOS-Endovis18/valid/JPEGImages --gt_root ./datasets/VOS-Endovis18/valid/Annotations --gpu_id 0
