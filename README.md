# Roadline Semantic Segmentation
Semantic segmentation model based on U-Net with ResNet50 encoder.

## 🗃️ Training Data
Training data consists of **202 images** which are frames from **dash cameras videos**. Labeling was performed using the Supervisely service.

## ❓ How to use
This guide is for **Linux**, however project works in **Windows 10** and higher as well, just make sure you have **PyTorch** and **CUDA** installed in your system. 
**Python 3.11.x** is preferred.
### 1. Clone repository
```
git clone https://github.com/dead4an/roadline-semantic-segmentation.git
cd roadline-semantic-segmentation
```
### 2. Install requirements

**Linux**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
**Windows**
```
python3 -m venv .venv
./.venv/scripts/activate
pip install -r requirements.txt
```
### 3. Run train_model.py script
```
python3 train_model.py
```
In case of troubles due to **CUDA out of memory** I recommend to change **BATCH_SIZE** parameter in **train_model.py** to lower values,
also you can lower **resize_shape** parameter in **train_dataset** initialization.

### 4. Inference
You can find example of inference in [inference.ipynb](inference.ipynb).

## 💡 Improvement ideas
- Extend training dataset
- Add augmentation
- Test other encoders
