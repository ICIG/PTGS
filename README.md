# PTGS
## 📦 Installation
We strongly recommend using **Conda** to build the running environment, the environment configuration is based on the optimized `environment.yml` for Gaussian Splatting and talking head generation, which includes PyTorch3D, CUDA toolkit and other core dependencies.

```bash
git clone https://github.com/ICIG/PTGS.git
cd PTGS
# Initialize and update submodules (Gaussian Splatting related)
git submodule update --init --recursivez

# Create and activate conda environment
conda env create -f environment.yml
conda activate ptgs
```

Install optional dependencies (if not already included):

```bash
# Install custom gaussian rasterization
pip install -e submodules/custom-gaussian-rasterization
# Install simple knn for Gaussian Splatting
pip install -e submodules/simple-knn
#Install gridencoder
pip install -e gridencoder
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
```
- Prepare the environment for [EasyPortrait](https://github.com/hukenovs/easyportrait):

  ```bash
  # prepare mmcv
  conda activate talking_gaussian
  pip install -U openmim
  mim install mmcv-full==1.7.1

  # download model weight
  cd data_utils/easyportrait
  wget "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/experiments/models/fpn-fp-512.pth"
  ```



## 🛠️ Usage

### Pre-processing Training Video

* Put training video under `data/<ID>/<ID>.mp4`.

  The video **must be 25FPS, with all frames containing the talking person**. 
  The resolution should be about 512x512, and duration about 1-5 min.

* Run script to process the video.

  ```bash
  python data_utils/process.py data/<ID>/<ID>.mp4
  ```

* Obtain Action Units
  
  Run `FeatureExtraction` in [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), rename and move the output CSV file to `data/<ID>/au.csv`.

* Generate tooth masks

  ```bash
  export PYTHONPATH=./data_utils/easyportrait 
  python ./data_utils/easyportrait/create_teeth_mask.py ./data/<ID>
  ```

### Audio Pre-process

In our paper, we use DeepSpeech features for evaluation. 

* DeepSpeech

  ```bash
  python data_utils/deepspeech_features/extract_ds_features.py --input data/<name>.wav # saved to data/<name>.npy
  ```


### Train

```bash
# If resources are sufficient, partially parallel is available to speed up the training. See the script.
bash scripts/train_xx.sh data/<ID> output/<project_name> <GPU_ID>
```
### Test

```bash
# saved to output/<project_name>/test/ours_None/renders
python synthesize_fuse.py -S data/<ID> -M output/<project_name> --eval  
```

### Inference with Specified Audio

```bash
python synthesize_fuse.py -S data/<ID> -M output/<project_name> --use_train --audio <preprocessed_audio_feature>.npy
```
---
