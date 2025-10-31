# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time
from tracemalloc import start

sys.path.append("/home/gxu/proj1/lesionSeg/utswtumor_track")
sys.path.append("/home/gxu/proj1/lesionSeg/utswtumor_track/SurgSAM2_Mem")
sys.path.append("/home/gxu/proj1/lesionSeg/utswtumor_track/SurgSAM2_Mem/sam2")

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import json
from PIL import Image
from sympy import evaluate

from sam2.build_sam import build_sam2_video_predictor
from sav_dataset.utils.endo_sav_benchmark import benchmark

# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if not os.path.exists(input_mask_path):
            pass
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    output_per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not output_per_obj_png_file:
        os.makedirs(os.path.join(output_mask_dir, video_name, 'all'), exist_ok=True)
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, 'all', f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask[output_mask>0] = object_id
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_separate_inference_per_object(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    read_frame_interval=1,
    save_frame_interval=1,
):
    """
    Run VOS inference on a single video with the given predictor.

    Unlike `vos_inference`, this function run inference separately for each object
    in a video, which could be applied to datasets like LVOS or YouTube-VOS that
    don't have all objects to track appearing in the first frame (i.e. some objects
    might appear only later in the video).
    """
    # load the video frames and initialize the inference state on this video
    video_dir = os.path.join(base_video_dir, video_name)
    all_frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", '.png']
    ]
    # only process frames with frame_interval
    frame_names = [p for p in all_frame_names if int(os.path.splitext(p)[0]) % read_frame_interval == 0]
    # save_pred_freq = len(all_frame_names) // len(frame_names)
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    # collect all the object ids and their input masks
    inputs_per_object = defaultdict(dict)
    for idx, name in enumerate(frame_names):
        if os.path.exists(os.path.join(input_mask_dir, video_name, f"{name}.png")):
            per_obj_input_mask, input_palette = load_masks_from_dir(
                input_mask_dir=input_mask_dir,
                video_name=video_name,
                frame_name=frame_names[idx],
                per_obj_png_file=False,  # our dataset combines all object masks into a single PNG file
                allow_missing=False,
            )
            for object_id, object_mask in per_obj_input_mask.items():
                # skip empty masks
                if not np.any(object_mask):
                    continue
                # if `use_all_masks=False`, we only use the first mask for each object
                if len(inputs_per_object[object_id]) > 0 and not use_all_masks:
                    continue
                print(f"adding mask from frame {idx} as input for {object_id=}")
                inputs_per_object[object_id][idx] = object_mask

    # # step 1: run inference together for each object appearing in the first frame
    # object_ids = sorted(inputs_per_object)
    # output_scores_per_object = defaultdict(dict)
    # # find the object appear in the first frame
    # first_frame_idx = 0
    # first_frame_object_ids = []
    # latter_frame_object_ids = []
    # for object_id in object_ids:
    #     if inputs_per_object[object_id].keys().__contains__(first_frame_idx):
    #         first_frame_object_ids.append(object_id)
    #     else:
    #         latter_frame_object_ids.append(object_id)
    #
    # for object_id in first_frame_object_ids:
    #     predictor.add_new_mask(
    #         inference_state=inference_state,
    #         frame_idx=first_frame_idx,
    #         obj_id=object_id,
    #         mask=inputs_per_object[object_id][first_frame_idx],
    #     )
    #
    # # run propagation throughout the video and collect the results in a dict
    # for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
    #     inference_state, start_frame_idx=first_frame_idx, reverse=False,
    # ):
    #     obj_scores = out_mask_logits.cpu().numpy()
    #     for i, out_obj_id in enumerate(out_obj_ids):
    #         output_scores_per_object[out_obj_id][out_frame_idx] = obj_scores[i: i + 1]

    # step 2: run inference separately for the object appearing in the latter frame
    # for object_id in latter_frame_object_ids:
    object_ids = sorted(inputs_per_object)
    output_scores_per_object = defaultdict(dict)
    fps_seq_all = []
    for object_id in object_ids:
        # add those input masks to SAM 2 inference state before propagation
        input_frame_inds = sorted(inputs_per_object[object_id])
        predictor.reset_state(inference_state)
        for input_frame_idx in input_frame_inds:
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=inputs_per_object[object_id][input_frame_idx],
            )

        # run propagation throughout the video and collect the results in a dict
        start_time = time.time()  # start the timer for inference time calculation
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=min(input_frame_inds),
            reverse=False,
        ):  
            ## calculate the inference time for each frame
            obj_scores = out_mask_logits.cpu().numpy()
            output_scores_per_object[object_id][out_frame_idx] = obj_scores
            ## calculate frame per second (FPS)
        end_time = time.time()  # end the timer for inference time calculation
        inference_time = end_time - start_time
        fps_seq = len(frame_names) / inference_time if inference_time > 0 else 0
        fps_seq_all.append(fps_seq)
        print(f"FPS for object {object_id}: {fps_seq:.2f} frames per second")
        # print(f"Processing frame {out_frame_idx} for object {object_id}, inference time: {inference_time:.2f} seconds")  

    print(f"Average FPS for all objects: {np.mean(fps_seq_all):.2f} frames per second")


    video_segments = {}
    # save_frame_interval，控制保存频率
    for out_frame_idx in range(len(frame_names)):
        frame_time = int(frame_names[out_frame_idx])
        if frame_time % save_frame_interval != 0:
            continue
        video_segments[out_frame_idx] = {}
        for object_id in object_ids:
            if output_scores_per_object[object_id].keys().__contains__(out_frame_idx):
                video_segments[out_frame_idx][object_id] = output_scores_per_object[object_id][out_frame_idx] > score_thresh

    # step 3: save the output masks as per-object PNG files
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            per_obj_output_mask=per_obj_output_mask,
            frame_name=frame_names[out_frame_idx],
            output_per_obj_png_file=True,
            height=height,
            width=width,
            output_palette=input_palette,
        )

    # step 4: save the output masks as a single PNG file
    # post-processing: consolidate the per-object scores into per-frame masks
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for frame_idx in range(len(frame_names)):
        frame_time = int(frame_names[frame_idx])
        if frame_time % save_frame_interval != 0:
            continue
        scores = torch.full(
            size=(len(object_ids), 1, height, width),
            fill_value=-1024.0,
            dtype=torch.float32,
        )
        for i, object_id in enumerate(object_ids):
            if frame_idx in output_scores_per_object[object_id]:
                scores[i] = torch.from_numpy(
                    output_scores_per_object[object_id][frame_idx]
                )

        scores = predictor._apply_non_overlapping_constraints(scores)
        per_obj_output_mask = {
            object_id: (scores[i] > score_thresh).cpu().numpy()
            for i, object_id in enumerate(object_ids)
        }
        video_segments[frame_idx] = per_obj_output_mask

    # write the output masks as palette PNG files to output_mask_dir
    for frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            output_per_obj_png_file=False,
            output_palette=input_palette,
        )

    return np.mean(fps_seq_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        # default="configs/sam2.1/sam2.1_hiera_t512_medsam2.yaml",        
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default= "/home/gxu/proj1/lesionSeg/utswtumor_track/SurgSAM2_Mem/sam2_logs/endovis2017_surgsam2_ts_ms_prune_cosine_sp/checkpoints/checkpoint_30.pt", # "./checkpoints/sam2.1_hiera_small.pt", "./sam2_logs/surgsam2_ori/checkpoints/checkpoint_30.pt"
        # default= "/home/gxu/proj1/lesionSeg/utswtumor_track/SurgSAM2_Mem/checkpoints/sam2.1_hiera_small.pt", # "./checkpoints/sam2.1_hiera_small.pt", "./sam2_logs/surgsam2_ori/checkpoints/checkpoint_30.pt"
        # default= "./sam2_logs/surgsam2_ori/checkpoints/checkpoint_30.pt",
        # default= "/home/gxu/proj1/lesionSeg/utswtumor_track/MedSAM2_new/checkpoints/medsam2/MedSAM2_latest.pt", # "./checkpoints/sam2.1_hiera_small.pt", "./sam2_logs/surgsam2_ori/checkpoints/checkpoint_30.pt"
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        default="/home/gxu/proj1/lesionSeg/utswtumor_track/SurgSAM2_Mem/sam2_logs/endovis2017_temp/results",
        # default="/home/gxu/proj1/lesionSeg/utswtumor_track/SurgSAM2_Mem/sam2_logs/endovis2017_SAM_Ori_with_prune/results",
        # default="/home/gxu/proj1/lesionSeg/utswtumor_track/SurgSAM2_Mem/sam2_logs/endovis2017_medsam2_ori/results",
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        default="/home/gxu/proj1/lesionSeg/utswtumor_track/SurgSAM2_Mem/datasets/VOS-Endovis17/valid/JPEGImages",
        help="directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument(
        "--input_mask_dir",
        type=str,
        default="/home/gxu/proj1/lesionSeg/utswtumor_track/SurgSAM2_Mem/datasets/VOS-Endovis17/valid/VOS/Annotations_vos_instrument",
        help="directory containing input masks (as PNG files) of each video",
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="text file containing the list of video names to run VOS prediction on",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--use_all_masks",
        default=False,
        help="whether to use all available PNG files in input_mask_dir "
        "(default without this flag: just the first PNG file as input to the SAM 2 model; "
        "usually we don't need this flag, since semi-supervised VOS evaluation usually takes input from the first frame only)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        default=False,
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    # we track the object appear in the first frame and the object appear in the latter frame
    parser.add_argument(
        "--first_frame_prompt_file",
        type=str, default=None,
    )
    parser.add_argument(
        "--gt_root",
        default="/home/gxu/proj1/lesionSeg/utswtumor_track/SurgSAM2_Mem/datasets/VOS-Endovis17/valid/Annotations",
        help="Path to the GT folder. For SA-V, it's sav_val/Annotations_6fps or sav_test/Annotations_6fps",
    )
    parser.add_argument(
        "--gpu_id",
        type=int, default=1,
    )
    parser.add_argument(
        "--read_frame_interval",
        type=int, default=1,
    )
    parser.add_argument(
        "--save_frame_interval",
        type=int, default=1,
    )
    # ----
    parser.add_argument(
        "-n", "--num_processes", default=16, type=int, help="Number of concurrent processes"
    )
    parser.add_argument(
        "-s",
        "--strict",
        help="Make sure every video in the gt_root folder has a corresponding video in the prediction",
        default=False,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="Quietly run evaluation without printing the information out",
        default=False,
    )
    parser.add_argument(
        "--do_not_skip_first_and_last_frame",
        help="In SA-V val and test, we skip the first and the last annotated frames in evaluation. "
             "Set this to true for evaluation on settings that doen't skip first and last frames",
        default=False,
    )
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    print(f"Using GPU {args.gpu_id}")
    print('Warning: only support evaluating one object per sequence and saving in object directories.')
    print('Testing with first frame prompt file:', args.first_frame_prompt_file)

    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=[],
    )

    if args.use_all_masks:
        print("using all available masks in input_mask_dir as input to the SAM 2 model")
    else:
        print(
            "using only the first frame's mask in input_mask_dir as input to the SAM 2 model"
        )
    # if a video list file is provided, read the video names from the file
    # (otherwise, we use all subdirectories in base_video_dir)
    if args.video_list_file is not None:
        with open(args.video_list_file, "r") as f:
            video_names = [v.strip() for v in f.readlines()]
    else:
        video_names = [
            p
            for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
               and int(p.split('_')[1]) in [9, 10] # 
        ]
    print(f"running VOS prediction on {len(video_names)} videos:\n{video_names}")

    # we first run every object separately and then combine them
    fps_mean = []
    for n_video, video_name in enumerate(video_names):
        # if n_video >= 5:
        #     continue
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        fps_mean_one_video = vos_separate_inference_per_object(
            predictor=predictor,
            base_video_dir=args.base_video_dir,
            input_mask_dir=args.input_mask_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            score_thresh=args.score_thresh,
            use_all_masks=args.use_all_masks,
            read_frame_interval=args.read_frame_interval,
            save_frame_interval=args.save_frame_interval,
        )
        fps_mean.append(fps_mean_one_video)
    fps_mean = np.mean(fps_mean)
    print(f"Average FPS for Total videos: {fps_mean:.2f} frames per second")

        # write into txt file
    with open(os.path.join(args.output_mask_dir, 'fps_sam2.txt'), 'a') as f:
        f.write(f"Average FPS for all objects: {np.mean(fps_mean):.2f} frames per second\n")


    print(
        f"completed VOS prediction on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )

    epoch = args.sam2_checkpoint.split('/')[-1].split('.')[0].split('_')[-1]
    epoch = int(epoch) if epoch.isdigit() else 0

    benchmark(
        [args.gt_root],
        [args.output_mask_dir],
        args.strict,
        args.num_processes,
        verbose=not args.quiet,
        skip_first_and_last=not args.do_not_skip_first_and_last_frame,
        epoch=epoch,
    )



if __name__ == "__main__":
    main()
