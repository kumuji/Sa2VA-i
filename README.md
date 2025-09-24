# Sa2VA-i: Improving Sa2VA Results with Consistent Training and Inference

[![Paper](https://img.shields.io/badge/arXiv-2509.19082-b31b1b.svg)](https://arxiv.org/abs/2509.19082)
[![GitHub](https://img.shields.io/badge/GitHub-Code-success)](https://github.com/kumuji/sa2va-i)

**3rd Place Report of LSVOS 2025 MeViS Track**

**[Alexey Nekrasov](https://scholar.google.com/citations?user=xJW2v3cAAAAJ)**<sup>1</sup> · **[Ali Athar](https://scholar.google.com/citations?user=mexenQMAAAAJ)** · **[Daan de Geus](https://scholar.google.com/citations?user=4gX3HRoAAAAJ)**<sup>2</sup> · **[Alexander Hermans](https://scholar.google.com/citations?user=V0iMeYsAAAAJ)**<sup>1</sup> · **[Bastian Leibe](https://scholar.google.com/citations?user=ZcULDB0AAAAJ)**<sup>1</sup>

<sup>1</sup>RWTH Aachen University · <sup>2</sup>Eindhoven University of Technology

![Teaser](https://arxiv.org/html/2509.19082v1/x1.png)

## 🚀 Overview

Sa2VA-i is an improved version of the popular **Sa2VA** model for language-guided dense grounding in images and video.
While Sa2VA achieves state-of-the-art results on multiple segmentation benchmarks, we identified critical inconsistencies between training and inference procedures that limited its full potential for referring video object segmentation tasks.

**Key improvements in Sa2VA-i:**
- ✅ **Consistent training and inference** - eliminates incompatibility between finetuned mask decoder and frozen memory components of SAM2
- ✅ **Improved frame sampling** - uniform sampling instead of first-frame sampling
- ✅ **Better mask propagation** - uses original SAM2 weights for propagation while keeping finetuned decoder for initial predictions
- ✅ **Significant performance gains** - up to +11.6 J&F on MeViS, +1.4 on Ref-YT-VOS, +3.3 on Ref-DAVIS

## 📊 Performance Highlights

| Model | MeViS (J&F) | Ref-YT-VOS (J&F) | Ref-DAVIS17 (J&F) |
|-------|-------------|------------------|-------------------|
| Sa2VA-1B | 47.0 | 68.0 | 69.5 |
| **Sa2VA-i-1B** | **52.6** | **70.3** | **73.6** |
| Sa2VA-4B | 46.4 | 71.3 | 73.7 |
| **Sa2VA-i-4B** | **56.6** | **73.2** | **78.6** |
| Sa2VA-8B | 51.5 | 72.3 | 75.9 |
| **Sa2VA-i-8B** | **59.5** | **73.9** | **79.1** |
| Sa2VA-26B | 52.1 | 75.1 | 78.6 |
| **Sa2VA-i-26B** | **63.2** | **76.5** | **81.2** |

**Note:** Sa2VA-i-1B performs on par with original Sa2VA-26B on MeViS benchmark!

## 🏆 Competition Results

**3rd Place** in LSVOS 2025 MeViS Track (RVOS) with **64.1 J&F**

## 🤗 Model Zoo

Sa2VA-i provides improved inference procedures for existing Sa2VA models. Available models:

| Model | HuggingFace Repository |
|-------|------------------------|
| Sa2VA-i-1B | [kumuji/Sa2VA-i-1B](https://huggingface.co/kumuji/Sa2VA-i-1B) |
| Sa2VA-i-4B | [kumuji/Sa2VA-i-4B](https://huggingface.co/kumuji/Sa2VA-i-4B) |
| Sa2VA-i-8B | [kumuji/Sa2VA-i-8B](https://huggingface.co/kumuji/Sa2VA-i-8B) |
| Sa2VA-i-26B | [kumuji/Sa2VA-i-26B](https://huggingface.co/kumuji/Sa2VA-i-26B) |

## 🎯 Quick Start

For installation and basic usage, please refer to the original [Sa2VA repository](https://github.com/magic-research/Sa2VA).
Sa2VA-i is a drop-in replacement for inference.

## 🔧 Key Improvements

### 1. Consistent Training-Inference
Eliminates incompatibility between finetuned mask decoder and frozen memory components by using the same procedure during both training and inference.

### 2. Improved Frame Sampling
Replaces first-frame sampling with uniform sampling for better coverage of video content.

### 3. Original SAM2 Mask Propagation
Uses original SAM2 weights for propagation while keeping finetuned decoder for initial mask predictions.

## 📚 Citation

If you use Sa2VA-i in your research, please cite:

```bibtex
@article{sa2va2025improved,
  title={Sa2VA-i: Improving Sa2VA Results with Consistent Training and Inference},
  author={Nekrasov, Alexey and Athar, Ali and de Geus, Daan and Hermans, Alexander and Leibe, Bastian},
  journal={arXiv preprint arXiv:2509.19082},
  year={2025}
}
```

Shout-out to the original Sa2VA paper!
```bibtex
@article{yuan2025sa2va,
  title={Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos},
  author={Yuan, Haobo and Li, Xiangtai and Zhang, Tao and Huang, Zilong and Xu, Shilin and Ji, Shunping and Tong, Yunhai and Qi, Lu and Feng, Jiashi and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2501.04001},
  year={2025}
}
```
