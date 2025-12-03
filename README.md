# DeepFashion Attribute Classifier

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Multi-Task Deep Learning for Fashion Image Analysis**

[Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) • [Pre-trained Model](#testing-the-models) • [Documentation](#running-experiments)

</div>

---

![img](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/attributes.jpg)

## 📋 Overview

A PyTorch-based multi-task convolutional neural network for comprehensive fashion image analysis using the DeepFashion dataset. This framework simultaneously performs:
- **Object Localization** via bounding box prediction
- **Category Classification** across 50 clothing categories
- **Fine-Grained Attribute Recognition** for 1,000 fashion attributes

### Key Features

🎯 **Multi-Task Architecture**: Unified model for simultaneous category, attribute, and bounding box prediction

🔍 **Global-Local Feature Fusion**: Leverages predicted bounding boxes to extract localized texture details while maintaining global context

⚡ **Optimized Training**: NVIDIA Apex integration for mixed-precision (FP16) training, reducing memory footprint and accelerating throughput

📊 **Large-Scale Dataset**: Trained on 289,222 images with rich annotations

### Dataset Statistics

The [DeepFashion Category and Attribute Prediction Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) provides:
- **289,222** annotated clothing images
- **50** clothing categories
- **1,000** descriptive attributes
- Bounding box and category labels for each image

### Architecture

Our implementation incorporates global and local feature streams through a multi-branch architecture. The bounding box prediction module enables extraction of fine-grained local features, which are fused with global representations for improved classification performance.

<img src="https://user-images.githubusercontent.com/2417792/56816446-9086d900-6811-11e9-9afa-ce3787d50558.png" width=750 />

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended)
- Anaconda/Miniconda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/FarnooshGhadiri/Cloth_category_classifier.git
cd Cloth_category_classifier
```

2. **Set up conda environment**
```bash
conda env create -f environment.yaml
conda activate deepfashion
```

3. **[Optional] Install NVIDIA Apex for mixed-precision training**

Follow the installation instructions at [NVIDIA/apex](https://github.com/NVIDIA/apex). Once installed, enable FP16 training by adding `--fp16` to your training command.

### Download Dataset

1. Access the dataset from [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWGplNFhjc01NbzQ)
2. Extract to your preferred directory

---

## 🔧 Running Experiments

### Training

Train the model using `train.py`. Here's the configuration for our best-performing model:

```bash
python train.py \
--img_size=256 \
--crop_size=224 \
--num_workers=8 \
--data_dir=<path_to_data_dir> \
--batch_size=32 \
--name=<experiment_name> \
--reduce_sum \
--beta=0.001 \
--resume=auto \
--epochs=200 \
--data_aug \
--lr=2e-4
```

**Training Outputs**:
- Model statistics are logged to `<experiment_name>/results.txt` after each epoch
- Includes training/validation losses and accuracies

**Evaluation Modes**:
- `--mode=train` (default): Training mode
- `--mode=validate`: Validation set evaluation
- `--mode=test`: Test set evaluation

Run `python train.py --help` for complete parameter documentation.

### Testing

#### Download Pre-trained Model

```bash
bash download_pretrained.sh
```

This downloads a checkpoint to `results/pretrained_model/` (trained with the script above).

#### Single Image Inference

```bash
python test_on_image.py \
--df_dir=<path_to_deepfashion_dir> \
--checkpoint=<path_to_checkpoint> \
--filename=<path_to_image> \
--out_file=<path_to_output>
```

#### Batch Inference

Process multiple images at once:

```bash
# Example: Process all images in a category folder
IMGS=`find '/path/to/deep_fashion/dataset/img/Boxy_Pocket_Top' -path '*.jpg' | tr '\n' ','`
python test_on_image.py \
--df_dir=/path/to/deep_fashion/ \
--checkpoint=results/pretrained_model/epoch_340.pth \
--filenames="${IMGS}" \
--out_folder=predictions/
```

**Additional Options**:
- `--top_k`: Number of top attributes to display per category
- `--softmax_temp`: Temperature scaling for prediction confidence adjustment

Run `python test_on_image.py --help` for all options.

---

## ⚠️ Known Limitations

The dataset contains images with multiple clothing items (e.g., skirt + blouse) but only single-label annotations. This creates inherent ambiguity in multi-item scenarios.

**Dataset Structure**: The 50 categories are organized into three subcategories:
- **Label 1**: Topwear (shirts, blouses, etc.)
- **Label 2**: Bottomwear (skirts, pants, etc.)
- **Label 3**: Full-body (dresses, jumpsuits, etc.)

A more robust approach would employ separate classification branches for each subcategory rather than a single unified classifier.

---

## 🤝 Contributing

Issues and pull requests are welcome! Please file bug reports or feature requests in the [Issues](https://github.com/FarnooshGhadiri/Cloth_category_classifier/issues) section.

---

## 📄 License

This project is released under the MIT License.

---

## 🙏 Acknowledgments

This work builds upon the [DeepFashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) from The Chinese University of Hong Kong.

---

<div align="center">

**⭐ Star this repo if you find it useful!**

</div>