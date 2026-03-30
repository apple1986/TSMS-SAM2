# TSMS-SAM2  
**Multi-scale Temporal Sampling Augmentation and Memory-Splitting Pruning for Promptable Video Object Segmentation and Tracking in Surgical Scenarios**  
📄 [arXiv:2508.05829](https://arxiv.org/abs/2508.05829)  
📦 [GitHub Repository](https://github.com/apple1986/TSMS-SAM2)

---

## 🚀 Overview
**TSMS-SAM2** is a novel framework for **promptable video object segmentation and tracking (VOST)** in surgical videos, built on the foundation of **Segment Anything Model 2 (SAM2)**.  
The framework addresses two critical challenges in surgical scenarios:
- **Rapid, irregular object motion** across video frames.
- **Redundancy in memory features** that slows down inference and increases computational load.

To overcome these issues, **TSMS-SAM2** introduces two key innovations:
1. **Multi-scale Temporal Sampling Augmentation (TS):** dynamically samples training frames at multiple temporal intervals to enhance robustness across various motion speeds.
2. **Memory-Splitting & Pruning (MS):** divides stored memory features into short-term and long-term groups and prunes redundant features to maintain efficiency and performance.

---

## ✨ Key Contributions
- 🧠 A **promptable, SAM2-based video segmentation framework** tailored for surgical videos.
- 🎞️ **Multi-scale temporal sampling** strategy for robust motion understanding.
- 🧩 **Memory-splitting and pruning** for efficient memory management and inference.
- 📈 **Superior accuracy and efficiency** on benchmark datasets (EndoVis2017, EndoVis2018).
- 🔬 Comprehensive **ablation studies** confirming the effectiveness of each module.

---

## 🧬 Architecture Overview
The TSMS-SAM2 framework extends the SAM2 backbone with:
- **Temporal Sampling Module (TS):** to diversify temporal intervals during training.
- **Memory-Splitting Unit (MSU):** separates recent and historical memory embeddings.
- **Pruning Gate:** removes redundant feature vectors based on cosine similarity.
- **Promptable decoder:** supports mask, box, and point prompts for flexible inference.

<div align="center">
<img src="docs/tsms_sam2_framework.png" width="700">
</div>

---

## 📊 Experimental Results
| Dataset | Metric | SAM2 | SAM2-TS | SAM2-MS | **TSMS-SAM2 (Ours)** |
|:---------|:--------|:------|:---------|:---------|:------------------:|
| **EndoVis2017** | Dice (%) | 92.31 | 93.78 | 94.22 | **95.24 ± 0.96** |
| **EndoVis2018** | Dice (%) | 83.02 | 84.61 | 85.20 | **86.73 ± 15.46** |

- TSMS-SAM2 achieves consistent improvements in both accuracy and efficiency.  
- Memory usage reduced by **~30%** while maintaining high segmentation fidelity.

For detailed results and ablations, refer to the [paper](https://arxiv.org/pdf/2508.05829).

---

## 🗂 Repository Structure
```
TSMS-SAM2/
├── configs/             # Configuration files (training/inference)
├── data/                # Dataset preparation scripts
├── docs/                # Documentation, diagrams, figures
├── eval/                # Evaluation metrics and benchmarking
├── inference/           # Inference and tracking scripts
├── models/              # Model definitions (SAM2 + TSMS modules)
├── training/            # Training scripts and utilities
├── utils/               # Helper functions and modules
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone this repository
```bash
git clone https://github.com/apple1986/TSMS-SAM2.git
cd TSMS-SAM2
```

### 2️⃣ Create the environment
```bash
conda create -n tsms-sam2 python=3.10 -y
conda activate tsms-sam2
```

### 3️⃣ Install and set up SAM2 backbone  
Follow the [official SAM2 installation guide](https://github.com/facebookresearch/sam2)  
and place pretrained SAM2 weights in the `checkpoints/` directory.

---

## 🎯 Usage

### 🧩 Training
```bash
python training/train_tsms_sam2.py     --config configs/tsms_sam2_surg.yaml     --epochs 100 --batch_size 8
```

**Key Configs**
- `temporal_scales`: list of frame intervals for multi-scale sampling.  
- `memory_bank_size`: maximum stored memory features.  
- `pruning_threshold`: cosine similarity threshold for pruning.  

---

### 🔍 Inference & Tracking
```bash
python inference/predict_video.py     --video_path ./samples/surgery.mp4     --checkpoint ./checkpoints/tsms_sam2_best.pth     --prompt_type mask
```
Supported prompts:
- `mask`
- `box`
- `point`

---

### 🧪 Evaluation
```bash
python eval/evaluate.py     --pred_dir ./results/     --gt_dir ./datasets/EndoVis2017/ground_truth/
```

---

## 📈 Performance Highlights
- **EndoVis2017:** Dice = 95.24% ± 0.96  
- **EndoVis2018:** Dice = 86.73% ± 15.46  
- **Memory compression:** reduces memory by ≈ 30%  
- **Inference speed:** up to 1.4× faster than SAM2 baseline  

<div align="center">
<img src="docs/qualitative_results.png" width="700">
</div>

---

## ⚠️ Limitations & Future Work
- Temporal sampling currently uses fixed scales (e.g., ×1, ×2).  
  Future work will explore **adaptive temporal sampling** guided by motion cues.  
- Memory pruning parameters are manually set; dynamic thresholding could further enhance efficiency.  
- Evaluation limited to surgical videos — extending to general medical and natural videos is promising.

---

## 📚 Citation
If you use **TSMS-SAM2** in your research, please cite:
```bibtex
@article{xu2025tsms_sam2,
  title   = {TSMS-SAM2: Multi-scale Temporal Sampling Augmentation and Memory-Splitting Pruning for Promptable Video Object Segmentation and Tracking in Surgical Scenarios},
  author  = {Xu, Guoping and Shao, Hua-Chieh and Zhang, You},
  journal = {arXiv preprint arXiv:2508.05829},
  year    = {2025}
}
```

---

## 🙏 Acknowledgements
This work builds upon the **Segment Anything Model 2 (SAM2)** from Meta AI and leverages insights from the **EndoVis surgical video datasets**.  
We thank the open-source community for their contributions to video segmentation research. 
- Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning
- https://github.com/jinlab-imvr/Surgical-SAM-2

---

## 💫 License
This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## ⭐ Contributing
We welcome contributions!  
If you find this project helpful:
1. ⭐ Star the repo  
2. 🐛 Open an issue for bugs or feature requests  
3. 🔧 Submit a pull request for improvements

---

### 📬 Contact
For questions or collaboration opportunities, please reach out via the [GitHub Issues](https://github.com/apple1986/TSMS-SAM2/issues) page or email the corresponding author listed in the [paper] (https://iopscience.iop.org/article/10.1088/3049-477X/ae291a/meta) or (https://arxiv.org/abs/2508.05829).

---

<p align="center">
  <b>TSMS-SAM2 — Enhancing SAM2 for Temporal Understanding in Surgical Video Segmentation</b><br>
  <em>Developed by Guoping Xu, Hua-Chieh Shao, and You Zhang (2025)</em>
</p>

## Citation
Please cite it if it helps your project:
Xu, Guoping, Hua-Chieh Shao, and You Zhang. "TSMS-SAM2: multi-scale temporal sampling augmentation and memory-splitting pruning for promptable video object segmentation and tracking in surgical scenarios." Machine Learning: Health 2.1 (2026): 015003.
