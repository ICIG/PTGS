# PTGS
## 📦 Installation
We strongly recommend using **Conda** to build the running environment, the environment configuration is based on the optimized `environment.yml` for Gaussian Splatting and talking head generation, which includes PyTorch3D, CUDA toolkit and other core dependencies.

```bash
git clone https://github.com/ICIG/PTGS.git
cd PTGS
# Initialize and update submodules (Gaussian Splatting related)
git submodule update --init --recursivez

# Create and activate conda environment
conda env create -f egstalker.yml
conda activate egstalker
```

Install optional dependencies (if not already included):

```bash
# Install custom gaussian rasterization
pip install -e submodules/custom-gaussian-rasterization
# Install simple knn for Gaussian Splatting
pip install -e submodules/simple-knn
```
## 📅 Download Dataset

We use talking portrait videos (3–5 minutes) from:

- [AD-NeRF](https://github.com/YudongGuo/AD-NeRF)
- [GeneFace](https://github.com/yerfor/GeneFace)
- [HDTF dataset](https://github.com/MRzzm/HDTF)

Example download:

```bash
wget https://github.com/YudongGuo/AD-NeRF/blob/master/dataset/vids/Obama.mp4?raw=true -O data/obama/obama.mp4
```
## 🧾 Data Preparation

### 1. Face Parsing

```bash
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O data_utils/face_parsing/79999_iter.pth
```

### 2. 3D Morphable Model

Download Basel Face Model 2009 from [here](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details), and place `01_MorphableModel.mat` in:

```
data_utils/face_tracking/3DMM/
```

Then run:

```bash
cd data_utils/face_tracking
python convert_BFM.py
cd ../../
python data_utils/process.py ${YOUR_DATASET_DIR}/${DATASET_NAME}/${DATASET_NAME}.mp4 
```


## 🛠️ Usage

To train the model:

```bash
python train.py -s ${YOUR_DATASET_DIR}/${DATASET_NAME} \
                --model_path ${YOUR_MODEL_DIR} \
                --configs arguments/args.py
```
Rendering：
```bash
python render.py -s ${YOUR_DATASET_DIR}/${DATASET_NAME} \
                 --model_path ${YOUR_MODEL_DIR} \
                 --configs configs/egstalker_default.py \
                 --iteration 10000 \
                 --batch 16
```
Inference with Custom Audio:

Place `<custom_aud>.wav` and `<custom_aud>.npy` in `${YOUR_DATASET_DIR}/${DATASET_NAME}` and run:

```bash
python render.py -s ${YOUR_DATASET_DIR}/${DATASET_NAME} \
                 --model_path ${YOUR_MODEL_DIR} \
                 --configs configs/egstalker_default.py \
                 --iteration 10000 \
                 --batch 16 \
                 --custom_aud <custom_aud>.npy \
                 --custom_wav <custom_aud>.wav \
                 --skip_train \
                 --skip_test
```

---
