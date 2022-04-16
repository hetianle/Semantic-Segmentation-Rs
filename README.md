# Semantic Segmentation Rs

## 训练

#### 单进程单机单卡模式
````bash
python tools/train.py --cfg experiments/loveda/seg_unet_2048x2048_sgd_lr1e-2_wd5e-4_bs_2_epoch100.yaml
````
#### 分布式单机多卡模式
````
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --cfg experiments/loveda/seg_unet_2048x2048_sgd_lr1e-2_wd5e-4_bs_2_epoch100.yaml --launcher pytorch
````

## 测试
目前只支持单进程单机单卡模式，如果只跑预测，不指定 label-dir 参数即可
````
python tools/test.py --cfg experiments/loveda/seg_unet_2048x2048_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml --checkpoint experiments/loveda/seg_unet_2048x2048_sgd_lr1e-2_wd5e-4_bs_12_epoch484/checkpoint.pth.tar --img-dir /mnt/lustrenew2/hetianle/data/lovedata/raw/Val/Urban/images_png/ --label-dir /mnt/lustrenew2/hetianle/data/lovedata/raw/Val/Urban/masks_png/
````
