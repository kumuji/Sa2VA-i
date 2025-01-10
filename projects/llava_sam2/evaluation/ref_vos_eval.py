import argparse
from pathlib import Path
import concurrent.futures
import json
import os

import cv2
import mmengine
import numpy as np
import torch
import torch.distributed
import torch.utils.data
import tqdm
from PIL import Image
from pycocotools import mask as cocomask
from transformers import AutoModel, AutoTokenizer

from projects.llava_sam2.evaluation.dataset import RefVOSDataset
from projects.llava_sam2.evaluation.utils import (
    _init_dist_pytorch,
    _init_dist_slurm,
    collect_results_cpu,
    get_dist_info,
    get_rank,
)


def async_func(executor, func, **kwargs):
    future = executor.submit(func, **kwargs)
    return future


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(cocomask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]["counts"] = rle[-1]["counts"].decode()
    return rle


def mask_save(item, mask_prediction, save_dir, save_using_cv2=False):
    vid_id = item["video_id"]
    exp_id = item["exp_id"]
    save_path = os.path.join(save_dir, vid_id, exp_id)
    mmengine.mkdir_or_exist(save_path)
    for id_m, mask in enumerate(mask_prediction):
        if save_using_cv2:
            mask = (mask.astype(bool) * 255).astype(np.uint8)
            file_name = item["frames"][id_m]
            save_file = os.path.join(save_path, file_name + ".png")
            cv2.imwrite(save_file, mask)
        else:
            mask = Image.fromarray(mask.astype(np.float32) * 255).convert("L")
            file_name = item["frames"][id_m]
            save_file = os.path.join(save_path, file_name + ".png")
            mask.save(save_file)


DATASETS_INFO = {
    "DAVIS": {
        "data_root": "/davis17/",
        "image_folder": "/davis17/valid/JPEGImages/",
        "expression_file": "/davis17/meta_expressions/valid/meta_expressions.json",
        "mask_file": "/davis17/valid/mask_dict.pkl",
    },
    "MEVIS": {
        "data_root": "/mevis/valid/",
        "image_folder": "/mevis/valid/JPEGImages",
        "expression_file": "/mevis/valid/meta_expressions.json",
        "mask_file": None,
    },
    "MEVIS_U": {
        "data_root": "/mevis/valid_u/",
        "image_folder": "/mevis/valid_u/JPEGImages/",
        "expression_file": "/mevis/valid_u/meta_expressions.json",
        "mask_file": "/mevis/valid_u/mask_dict.json",
    },
    "REFYTVOS": {
        "data_root": "/youtube-vos/",
        "image_folder": "/youtube-vos/valid/JPEGImages/",
        "expression_file": "/youtube-vos/meta_expressions/valid/meta_expressions.json",
        "mask_file": None,
    },
    "REVOS": {
        "data_root": "/revos/REVOS/",
        "image_folder": "/revos/REVOS/JPEGImages/",
        "expression_file": "/revos/REVOS/meta_expressions_valid_.json",
        "mask_file": None,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="RefVOS")
    parser.add_argument("model_path", help="hf model path.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS_INFO.keys(),
        default="MEVIS",
        help="Specify a dataset",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)  # dummy

    parser.add_argument("--num_frames", type=int, default=5)  # dummy

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


if __name__ == "__main__":
    args = parse_args()

    print(args)
    save_path = args.save_path
    # language_data_path = "/nodes/cristal/work/nekrasov/data/language-data"
    language_data_path = args.data_path

    if args.launcher == "none":
        rank = 0
        world_size = 1
    elif args.launcher == "pytorch":
        _init_dist_pytorch("nccl")
        rank, world_size = get_dist_info()
    elif args.launcher == "slurm":
        _init_dist_slurm("nccl")
        rank, world_size = get_dist_info()

    model = (
        AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    dataset_info = DATASETS_INFO[args.dataset]

    image_folder = language_data_path + dataset_info["image_folder"]
    expression_file = language_data_path + dataset_info["expression_file"]
    mask_file = dataset_info["mask_file"]
    if mask_file is not None:
        mask_file = language_data_path + mask_file

    dataset = RefVOSDataset(
        image_folder=image_folder,
        expression_file=expression_file,
        mask_file=mask_file,
    )

    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=4,
        pin_memory=False,
        collate_fn=lambda x: x[0],
    )
    results = []
    executor = concurrent.futures.ThreadPoolExecutor()
    for item in tqdm.tqdm(dataloader):
        vid_id = item["video_id"]
        exp_id = item["exp_id"]
        if (Path(save_path) / vid_id / exp_id).exists():
            print(f"{(Path(save_path) / vid_id / exp_id)} exists.")
            continue
        with torch.no_grad():
            result = model.predict_forward(
                video=item["images"],
                text=item["text_prompt"],
                tokenizer=tokenizer,
                sample_num_frames=args.num_frames,
            )

        text_idx = 0
        text_prediction = result["prediction"]
        if len(result["prediction_masks"]) > 0:
            mask_prediction = result["prediction_masks"][text_idx]
        else:
            print(text_prediction)
            mask_prediction = np.zeros(
                (item["length"], item["ori_height"], item["ori_width"]), dtype=np.uint8
            )

        # only saving the frames that were sampled for prediction
        # if result.get("sample_idx") is not None:
        #     item["frames"] = np.array(item["frames"])[result["sample_idx"]].tolist()

        if args.submit:
            async_func(
                executor,
                mask_save,
                item=item,
                mask_prediction=mask_prediction,
                save_dir=save_path,
                save_using_cv2=args.dataset == "REFYTVOS",
            )
            # mask_save(item=item, mask_prediction=mask_prediction, save_dir=save_path)
            encoded_mask = None
        else:
            encoded_mask = mask_to_rle(mask_prediction)

        result = {
            "index": item["index"],
            "video_id": item["video_id"],
            "exp_id": item["exp_id"],
            "text_prediction": text_prediction,
            "frames": item["frames"],
            "exp": item["text_prompt"],
            "prediction_masks": encoded_mask,
        }
        results.append(result)

    executor.shutdown(wait=True)
    print(f"[Rank {rank}] : Finished.")

    if not args.submit:
        results = collect_results_cpu(results, len(dataset))
        if get_rank() == 0:
            final_results = {}
            for item in results:
                vid_id = item["video_id"]
                exp_id = item["exp_id"]
                if vid_id not in final_results:
                    final_results[vid_id] = {}
                assert exp_id not in final_results[vid_id]
                final_results[vid_id][exp_id] = item
            os.makedirs(save_path, exist_ok=True)
            json.dump(final_results, open(f"{save_path}/results.json", "w"))

    if rank == 0:
        print("Done")
