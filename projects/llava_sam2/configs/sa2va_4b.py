from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import AutoTokenizer
from xtuner.dataset import ConcatDataset
from xtuner.dataset.map_fns import template_map_fn_factory
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from projects.llava_sam2.datasets import (
    FlickrGCGDataset,
    GranDfGCGDataset,
    LLaVADataset,
    OpenPsgGCGDataset,
    OspreyDataset,
    OspreyDescriptionDataset,
    OspreyShortDescriptionDataset,
    RefCOCOgGCGDataset,
    ReferSegmDataset,
    VideoChatUniViDataset,
    VideoMeVISDataset,
    VideoRefYoutubeVOSDataset,
    VideoReVOSDataset,
    VideoSAM2Dataset,
    video_lisa_collate_fn,
)
from projects.llava_sam2.models import (
    SAM2TrainRunner,
    VideoLLaVASAMModel,
    VideoLLaVASAMModel_zero3,
)
from projects.llava_sam2.models.internvl import InternVL_Slowfast
from projects.llava_sam2.models.preprocess.image_resize import DirectResize
from third_parts.mmdet.models.losses import CrossEntropyLoss, DiceLoss

language_data_path = "/alexey/data/language-data"
coco_path = f"{language_data_path}/coco"
glamm_data_root = f"{language_data_path}/grandf"
osprey_path = f"{language_data_path}/Osprey-724K"
revos_path = f"{language_data_path}/revos/REVOS"
mevis_path = f"{language_data_path}/mevis/train"
refytvos_path = f"{language_data_path}/youtube-vos"
video_chatunivi_path = f"{language_data_path}/Chat-UniVi-Instruct/Fine-tuning/VIDEO"
llava_dataset_path = f"{language_data_path}/llava_dataset/LLaVA-Instruct-150K"

# Model
path = "/alexey/saved/sam2/InternVL2_5-4B"
pretrained_pth = None

#######################################################################
#                          PART 1  Settings                           #
#######################################################################

# Data
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 1
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
# official 1024 -> 4e-5
# lr = 1e-6
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

num_frames = 8
tarvis_num_frames = 8
num_tokens_per_expression = 1

# Save
save_steps = 1000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

special_tokens = ["[SEG]", "<p>", "</p>", "<vp>", "</vp>"]

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side="right",
)

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)
#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=VideoLLaVASAMModel_zero3,
    special_tokens=special_tokens,
    frozen_sam2_decoder=False,
    mllm=dict(
        type=InternVL_Slowfast,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        special_tokens=special_tokens,
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM2TrainRunner,
    ),
    loss_mask=dict(
        type=CrossEntropyLoss, use_sigmoid=True, reduction="mean", loss_weight=2.0
    ),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction="mean",
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5,
    ),
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    # loss_sample_points=False,
    bs=batch_size,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################


# VIDEO_DATAS = './data/video_datas/'
# IMG_DATAS = './data/image_datas/'

############### video res
video_revos_image_folder = f"{revos_path}/JPEGImages"
video_revos_expression_file = f"{revos_path}/meta_expressions_train_.json"
video_revos_mask_file = f"{revos_path}/mask_dict.json"

video_mevis_image_folder = f"{mevis_path}/JPEGImages"
video_mevis_expression_file = f"{mevis_path}/meta_expressions.json"
video_mevis_mask_file = f"{mevis_path}/mask_dict.json"

video_refytvos_image_folder = f"{refytvos_path}/train/JPEGImages/"
video_refytvos_expression_file = f"{refytvos_path}/meta_expressions/train/meta_expressions.json"
video_refytvos_mask_file = f"{refytvos_path}/mask_dict.pkl"

video_revos_dataset = dict(
    type=VideoReVOSDataset,
    image_folder=video_revos_image_folder,
    expression_file=video_revos_expression_file,
    mask_file=video_revos_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=10,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=num_frames,
    tarvis_sampled_frames=tarvis_num_frames,
    num_tokens_per_expression=num_tokens_per_expression,
)

video_mevis_dataset = dict(
    type=VideoMeVISDataset,
    image_folder=video_mevis_image_folder,
    expression_file=video_mevis_expression_file,
    mask_file=video_mevis_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=num_frames,
    tarvis_sampled_frames=tarvis_num_frames,
    num_tokens_per_expression=num_tokens_per_expression,
)

video_refytvos_dataset = dict(
    type=VideoRefYoutubeVOSDataset,
    image_folder=video_refytvos_image_folder,
    expression_file=video_refytvos_expression_file,
    mask_file=video_refytvos_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=num_frames,
    tarvis_sampled_frames=tarvis_num_frames,
    num_tokens_per_expression=num_tokens_per_expression,
)

################### Video chat
video_chatunivi_image_folder = f"{video_chatunivi_path}/Activity_Videos/"
video_chatunivi_json_file = f"{video_chatunivi_path}/video_chat.json"

video_qa_dataset = dict(
    type=VideoChatUniViDataset,
    image_folder=video_chatunivi_image_folder,
    json_file=video_chatunivi_json_file,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=num_frames,
)

################## image chat
llava_vqa_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path=f"{llava_dataset_path}/llava_v1_5_mix665k.json",
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    image_folder=f"{llava_dataset_path}/llava_images/",
)

################## image res
refcoco_segm_dataset = dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=f"{language_data_path}/refer_seg/refcoco",
    data_prefix=dict(img_path=f"{coco_path}/train2014/"),
    ann_file="instances.json",
    split_file="refs(unc).p",
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    num_tokens_per_expression=num_tokens_per_expression,
)
refcoco_plus_segm_dataset = dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=f"{language_data_path}/refer_seg/refcoco+",
    data_prefix=dict(img_path=f"{coco_path}/train2014/"),
    ann_file="instances.json",
    split_file="refs(unc).p",
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    num_tokens_per_expression=num_tokens_per_expression,
)
refcocog_segm_dataset = dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=f"{language_data_path}/refer_seg/refcocog",
    data_prefix=dict(img_path=f"{coco_path}/train2014/"),
    ann_file="instances.json",
    split_file="refs(umd).p",
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    num_tokens_per_expression=num_tokens_per_expression,
)

# image gcg datas
refcocog_image_path = f"{glamm_data_root}/coco_2014/train2014"
refcocog_ann_file = f"{glamm_data_root}/GranDf/annotations/train/RefCOCOg_GCG_train.json"

grandf_image_path = f"{glamm_data_root}/GranDf_HA_images/train/"
grandf_ann_file = f"{glamm_data_root}/GranDf/annotations/train/GranDf_HA_GCG_train.json"

flickr_image_path = f"{glamm_data_root}/flickr30k/flickr30k-images/"
flickr_ann_file = f"{glamm_data_root}/GranDf/annotations/train/flickr_mergedGT_GCG_train.json"

psg_image_path = f"{glamm_data_root}/coco_2017/"
psg_ann_file = f"{glamm_data_root}/GranDf/annotations/train/OpenPsgGCG_train.json"

glamm_refcocog_dataset = dict(
    type=RefCOCOgGCGDataset,
    image_folder=refcocog_image_path,
    data_path=refcocog_ann_file,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    num_tokens_per_expression=num_tokens_per_expression,
)

glamm_grandf_dataset = dict(
    type=GranDfGCGDataset,
    data_path=grandf_ann_file,
    image_folder=grandf_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=10,
    num_tokens_per_expression=num_tokens_per_expression,
)

glamm_psg_dataset = dict(
    type=OpenPsgGCGDataset,
    data_path=psg_ann_file,
    image_folder=psg_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    num_tokens_per_expression=num_tokens_per_expression,
)

glamm_flickr_dataset = dict(
    type=FlickrGCGDataset,
    data_path=flickr_ann_file,
    image_folder=flickr_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    num_tokens_per_expression=num_tokens_per_expression,
)

# # sam2 data
data_sam2_folder = '/alexey/data/sav'
data_sam2_expression_file = '/alexey/data/sav/Ref-SAV.json'

video_sam2_dataset = dict(
    type=VideoSAM2Dataset,
    sam2_folder=data_sam2_folder,
    expression_file=data_sam2_expression_file,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=num_frames,
    tarvis_sampled_frames=tarvis_num_frames,
    select_number=5,
    num_tokens_per_expression=num_tokens_per_expression,
)

# osprey
data_osprey_file = f"{osprey_path}/osprey_conversation.json"
data_osprey_image_folders = [
    f"{coco_path}/train2014/",
    f"{coco_path}/val2014/",
    f"{coco_path}/train2017/",
    f"{coco_path}/val2017/",
]

image_osprey_dataset = dict(
    type=OspreyDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_file,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

data_osprey_detail_description_file = f"{osprey_path}/osprey_detail_description.json"
image_osprey_description_dataset = dict(
    type=OspreyDescriptionDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_detail_description_file,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

data_osprey_short_file = f"{osprey_path}/osprey_short_form.json"
image_osprey_short_dataset = dict(
    type=OspreyShortDescriptionDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_short_file,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

data_osprey_part_file = f"{osprey_path}/osprey_part_level.json"
image_osprey_part_dataset = dict(
    type=OspreyDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_part_file,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

data_osprey_positive_neg_file = f"{osprey_path}/osprey_lvis_positive_negative.json"
image_osprey_positive_neg_dataset = dict(
    type=OspreyDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_positive_neg_file,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        # sem seg
        # semantic_seg_ade20k_dataset,
        # ref seg
        refcoco_segm_dataset,
        refcoco_plus_segm_dataset,
        refcocog_segm_dataset,
        refcoco_segm_dataset,
        refcoco_plus_segm_dataset,
        refcocog_segm_dataset,
        refcoco_segm_dataset,
        refcoco_plus_segm_dataset,
        refcocog_segm_dataset,
        refcoco_segm_dataset,
        refcoco_plus_segm_dataset,
        refcocog_segm_dataset,
        # image qa
        llava_vqa_dataset,
        # video res
        video_mevis_dataset,
        video_revos_dataset,
        video_refytvos_dataset,
        # video chat
        video_qa_dataset,
        # sam2 pesudo
        video_sam2_dataset,
        # gcg data
        glamm_psg_dataset,
        glamm_grandf_dataset,
        glamm_flickr_dataset,
        glamm_refcocog_dataset,
        # visual prompt
        image_osprey_dataset,
        image_osprey_description_dataset,
        image_osprey_part_dataset,
        image_osprey_short_dataset,
        image_osprey_positive_neg_dataset,
    ],
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property="modality_length",
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=video_lisa_collate_fn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="bfloat16",
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=True,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(
            type='TensorboardVisBackend',
            # save_dir='tensorboard_logs'  # Set the log folder to save_folder
        )
    ]
)

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = True

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=42, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
