## Image Super-Resolution with Deep Dictionary (ECCV 2022)

This repository provides the official PyTorch implementation of the following paper:<br>
Shunta Maeda, "Image Super-Resolution with Deep Dictionary", ECCV 2022.<br>

This repository is based on the official [CutBlur](https://github.com/clovaai/cutblur) repository.<br>
Other than the addition of the proposed model `model/srdd.py`, changes were made only to `solver.py` and `inference.py`.<br>

### Dataset
We use the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset to train the model. Download and unpack the tar file any directory you want.<br>
**Important:** For the DIV2K dataset only, all the train and valid images should be placed in the `DIV2K_train_HR` and `DIV2K_train_LR_bicubic` directories (We parse train and valid images using `--div2k_range` argument).

### Train
```shell
python main.py \
    --model SRDD \
    --dataset DIV2K_SR \
    --div2k_range 1-800/801-810 \
    --scale 4 \
    --dataset_root <directory_of_dataset> \
    --save_result \
    --patch_size 48 \
    --batch_size 32 \
    --lr 2e-4 \
    --decay "200-300-350-375" \
    --max_steps 400000
```

### Test
```shell
python inference.py \
    --model SRDD \
    --scale 4 \
    --pretrain <path_of_pretrained_model> \
    --dataset_root ./input \
    --save_root ./output
```

### Updates
- **14 July, 2022**: Initial upload.

### Acknowledgements
This code is built on [CutBlur](https://github.com/clovaai/cutblur). We thank the authors for sharing their codes.
