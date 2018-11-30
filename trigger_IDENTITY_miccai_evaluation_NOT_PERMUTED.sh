#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 python3.6 net_run.py inference -c train_IDENTITY_3D_HeMIS_single_task_Lesions_MICCAI_None.ini --app niftynet.contrib.pimms.3D_HeMIS_application.SegmentationApplication --dataset_split_file /home/tom/NiftyNet-fresh/NiftyNet/dataset_split_Lesions_MICCAI.csv --inference_iter 57900 --volume_padding_size 0,0,16 --save_seg_dir --permuted_indices 0,1,2 ./output/inference_iter_57900_idx_dropped_correct_splits_MICCAI_IDENTITY_None_NOT_PERMUTED
CUDA_VISIBLE_DEVICES=4 python3.6 net_run.py inference -c train_IDENTITY_3D_HeMIS_single_task_Lesions_MICCAI_None.ini --app niftynet.contrib.pimms.3D_HeMIS_application.SegmentationApplication --dataset_split_file /home/tom/NiftyNet-fresh/NiftyNet/dataset_split_Lesions_MICCAI.csv --inference_iter 57900 --volume_padding_size 0,0,16 --idxs_to_drop 0,1 --save_seg_dir --permuted_indices 0,1,2 ./output/inference_iter_57900_idx_dropped_correct_splits_MICCAI_IDENTITY_01_NOT_PERMUTED
CUDA_VISIBLE_DEVICES=4 python3.6 net_run.py inference -c train_IDENTITY_3D_HeMIS_single_task_Lesions_MICCAI_None.ini --app niftynet.contrib.pimms.3D_HeMIS_application.SegmentationApplication --dataset_split_file /home/tom/NiftyNet-fresh/NiftyNet/dataset_split_Lesions_MICCAI.csv --inference_iter 57900 --volume_padding_size 0,0,16 --idxs_to_drop 0,2 --save_seg_dir --permuted_indices 0,1,2 ./output/inference_iter_57900_idx_dropped_correct_splits_MICCAI_IDENTITY_02_NOT_PERMUTED
CUDA_VISIBLE_DEVICES=4 python3.6 net_run.py inference -c train_IDENTITY_3D_HeMIS_single_task_Lesions_MICCAI_None.ini --app niftynet.contrib.pimms.3D_HeMIS_application.SegmentationApplication --dataset_split_file /home/tom/NiftyNet-fresh/NiftyNet/dataset_split_Lesions_MICCAI.csv --inference_iter 57900 --volume_padding_size 0,0,16 --idxs_to_drop 1,2 --save_seg_dir --permuted_indices 0,1,2 ./output/inference_iter_57900_idx_dropped_correct_splits_MICCAI_IDENTITY_12_NOT_PERMUTED
CUDA_VISIBLE_DEVICES=4 python3.6 net_run.py inference -c train_IDENTITY_3D_HeMIS_single_task_Lesions_MICCAI_None.ini --app niftynet.contrib.pimms.3D_HeMIS_application.SegmentationApplication --dataset_split_file /home/tom/NiftyNet-fresh/NiftyNet/dataset_split_Lesions_MICCAI.csv --inference_iter 57900 --volume_padding_size 0,0,16 --idxs_to_drop 0 --save_seg_dir --permuted_indices 0,1,2 ./output/inference_iter_57900_idx_dropped_correct_splits_MICCAI_IDENTITY_0_NOT_PERMUTED
CUDA_VISIBLE_DEVICES=4 python3.6 net_run.py inference -c train_IDENTITY_3D_HeMIS_single_task_Lesions_MICCAI_None.ini --app niftynet.contrib.pimms.3D_HeMIS_application.SegmentationApplication --dataset_split_file /home/tom/NiftyNet-fresh/NiftyNet/dataset_split_Lesions_MICCAI.csv --inference_iter 57900 --volume_padding_size 0,0,16 --idxs_to_drop 1 --save_seg_dir --permuted_indices 0,1,2 ./output/inference_iter_57900_idx_dropped_correct_splits_MICCAI_IDENTITY_1_NOT_PERMUTED
CUDA_VISIBLE_DEVICES=4 python3.6 net_run.py inference -c train_IDENTITY_3D_HeMIS_single_task_Lesions_MICCAI_None.ini --app niftynet.contrib.pimms.3D_HeMIS_application.SegmentationApplication --dataset_split_file /home/tom/NiftyNet-fresh/NiftyNet/dataset_split_Lesions_MICCAI.csv --inference_iter 57900 --volume_padding_size 0,0,16 --idxs_to_drop 2 --save_seg_dir --permuted_indices 0,1,2 ./output/inference_iter_57900_idx_dropped_correct_splits_MICCAI_IDENTITY_2_NOT_PERMUTED
