### GroundedSAMLaMA

This repo combine Grounded-SAM: Detect and Segment Everything with Text Prompt, and LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions.
By using Grounded-SAM to get segments and masks for further use.
By using LaMa to inpaint. 

### Environment setup

Conda:

```bash
conda create --name GroundedSAMLaMA python=3.8
conda activate GroundedSAMLaMa
```


### Installation

Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

Install Grounding DINO:

```bash
python -m pip install -e GroundingDINO
```

Install LaMa:

```bash
python -m pip install torch torchvision
python -m pip install -r lama/requirements.txt
```

Remember to update requirements.txt before install!



### Grounded-SAM: Detect and Segment Everything with Text Prompt

**Step 1: Download the pretrained weights**

```bash
cd Grounded-Segment-Anything

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

**Step 2: Running grounded_sam_ian.py**

```python
export CUDA_VISIBLE_DEVICES=0
python grounded_sam_ian.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/e-bike.jpg \
  --output_dir "outputs" \
  --box_threshold 0.6 \
  --text_threshold 0.25 \
  --text_prompt "e-bike" \
  --device "cpu"
```
Change device to "cpu" if no cuda


### LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions

**Step 1: Download the pretrained weights**

```bash
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip
```

**Step 2: Running lama_inpaint_ian.py**

```python
lama_inpaint_ian.py \
  --input_img outputs/raw_image.png \
  --input_mask_glob "outputs/Mask/mask*.png" \
  --output_dir outputs \
  --lama_config ./lama/configs/prediction/default.yaml \
  --lama_ckpt ./big-lama
```
Ignore "Detectron v2 is not installed" for now


### Acknowledgements

- [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [LaMa](https://github.com/advimman/lama)


 
