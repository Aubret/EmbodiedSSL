
# 	Semantic Visual Representations Emerge from Embodied Self-Supervised Learning

Code repository for the paper "Semantic Visual Representations Emerge from Embodied Self-Supervised Learning"

## Code preparation

python -m pip install -r requirements.txt

## Dataset prepration

Download and uncompress the [Nymeria dataset](https://www.projectaria.com/datasets/nymeria/)

Run `python3 dataset_scripts/create_nymeria.py --data_root NYMERIA_PATH --fps 5 --resolution 512 --mode w`

## Pre-train models on Nymeria

Before running the code, modify the scripts to change the directories of the dataset and checkpoints (./scripts/pretrain/).
Furthermore, fill the wandb data to log the various metrics (wandb/private.yaml). By default, the code runs on 4 GPUS (++device=4). 

```
python3 main_pretrain.py --config-path scripts/pretrain/nymeria/ --config-name emocov3.yaml ++devices=4 ++data.train_path=NYMERIA_PATH
python3 main_pretrain.py --config-path scripts/pretrain/nymeria/ --config-name mocov3.yaml ++devices=4 ++data.train_path=NYMERIA_PATH
```

## Pre-trained models

The main pretrained models are provided [there](https://huggingface.co/aaubret/embodied_ssl/).

## Run evaluations

First download the datasets.

- For Imagenet: ./dataset_scritps/create_imgnetv2.py --data_root IMGNET_PATH --data_dest IMGNET_PATH
- For MvImgnet: we followed the preprocessing made in this [paper](https://arxiv.org/pdf/2407.06704?)
- For Places-365: use create_places365.py

For all scripts: change: ++data.train_path=PATH ++data.val_path=PATH ++checkpoint.dir=CHECKPOINT_PATH
```
python3 main_linear.py --config-path scripts/linear/nymeria/ --config-name imagenet.yaml ++data.dataset=imagenet2 ++pretrained_feature_extractor=PATH
python3 main_linear.py --config-path scripts/linear/nymeria/ --config-name nyu.yaml ++pretrained_feature_extractor=PATH
python3 main_linear.py --config-path scripts/linear/nymeria/ --config-name places.yaml ++pretrained_feature_extractor=PATH
python3 main_linear.py --config-path scripts/linear/nymeria/ --config-name mvimgnet.yaml ++pretrained_feature_extractor=PATH
python3 main_linear.py --config-path scripts/linear/nymeria/ --config-name segment.yaml ++pretrained_feature_extractor=PATH
python3 main_linear.py --config-path scripts/linear/nymeria/ --config-name davis.yaml ++pretrained_feature_extractor=PATH
```

To play with parameters:
- swap experiments: `++data.dataset_kwargs.swap_actions=True`
- Visuomotor sensitivity: `++method_kwargs.aa_weight=0.1`
- Temporal tuning: `++method_kwargs.time_window=5`
- Delayed locomotion: `++data.dataset_kwargs.delayed_actions=10 (epoch number)`
- Remove action dimensions: `++data.dataset_kwargs.mask=[0,0,1,1,1,1,1,1,1]`




## Figure generation

We provide the code to generate the figures in the notebook `figures_generation.ipynb`.

## Citation
To cite our [paper](https://www.researchsquare.com/article/rs-10140180/latest.pdf):
```bibtex
@article{aubret_semantic_2026,
	title = {Semantic {Visual} {Representations} {Emerge} from {Embodied} {Self}-{Supervised} {Learning}.},
	url = {https://www.researchsquare.com/article/rs-10140180/v1},
	author = {Aubret, Arthur and Teulière, Céline and Schaumlöffel, Timothy and Roig, Gemma and Triesch, Jochen},
	year = {2026},
}
```
