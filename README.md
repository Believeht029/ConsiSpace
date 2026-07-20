# ConsiSpace

### Learning Geometric Consistency Matters for Video Spatial Reasoning

[Ting Huang](https://believeht029.github.io/)<sup>2</sup>, [Zhenyu Zhang](https://jessezhang92.github.io/)<sup>1,†</sup>, Wenyuan Huang<sup>1</sup>, Jian Yang<sup>1</sup>, [Hao Tang](https://ha0tang.github.io/)<sup>2,3,†</sup>

<sup>1</sup> Nanjing University &nbsp; <sup>2</sup> Peking University &nbsp; <sup>3</sup> Beijing Academy of Artificial Intelligence
<sup>†</sup> Corresponding authors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Project Status](https://img.shields.io/badge/status-code%20coming%20soon-blue.svg)](#release-status)


## Overview

ConsiSpace is a geometry-consistency-aware framework for video spatial reasoning. It treats spatial consistency as both an evidence-organization principle and a post-SFT learning signal, enabling a multimodal model to reason more efficiently and reliably across long videos and changing viewpoints.

![Overview of the ConsiSpace framework](assets/structure.png)

The framework has two main components:

- **Geometry-Consistent Memory (GCM)** maintains complementary implicit and explicit spatial memories. Geometry-guided writing, fusion, and hierarchical retrieval reduce redundant observations while retaining query-relevant evidence.
- **Unified Consistency Self-Supervised Reinforcement Learning (UC-SSRL)** improves cross-view stability after supervised fine-tuning with answer-, metric-, and topology-consistency rewards, without requiring additional human annotations.

## Highlights

- Uses geometric consistency to control the complete memory lifecycle: **when to write, how to fuse, and what to retrieve**.
- Combines frozen **SigLIP2** visual features and **VGGT** geometry cues with **Qwen3-VL-8B-Instruct**.
- Updates only lightweight LoRA parameters during SFT and UC-SSRL.
- Improves the average score by **12.6 points** over the strongest baselines across three video spatial-reasoning benchmarks.
- Scales efficiently to long videos: at 200 frames, ConsiSpace uses **35.0 GB** and **3.10 s** per inference on an A100 80 GB, compared with **78.5 GB** and **10.80 s** for VLM-3R under the same setting.

## Main Results

| Benchmark | Setting | Strongest baseline | ConsiSpace (UC-SSRL) | Gain |
|---|---:|---:|---:|---:|
| VSI-Bench | Overall average | 69.6 | **76.6** | +7.0 |
| OSI-Bench | Overall average | 40.3 | **53.0** | +12.7 |
| MMSI-Video-Bench | Sufficient-Coverage | 42.6 | **57.5** | +14.9 |
| MMSI-Video-Bench | Uniform-50 | 43.1 | **58.1** | +15.0 |

VSI-Bench and OSI-Bench use Accuracy for multiple-choice tasks and Mean Relative Accuracy for numerical tasks. MMSI-Video-Bench reports exact-match accuracy.

## Training and Data

ConsiSpace is trained in two stages:

1. **Supervised fine-tuning:** instruction tuning on the indoor VSI-590K dataset and the outdoor nuScenes-10K dataset.
2. **UC-SSRL:** consistency refinement initialized from the SFT checkpoint using paired views of the same scene and question.

nuScenes-10K contains 10,000 spatial-reasoning QA pairs generated from nuScenes. The data pipeline filters 22,518 candidates using geometry validation, answer parsing, consistency checks, MLLM verification, and deduplication. Manual verification of 1,000 randomly sampled pairs found 91.6% correct annotations.

### Reference configuration

| Component | Configuration |
|---|---|
| Backbone | Qwen3-VL-8B-Instruct |
| Visual encoder | SigLIP2 (frozen) |
| Geometry encoder | VGGT (frozen) |
| Adaptation | LoRA, rank 16, alpha 32 |
| Memory | 64 tokens, dimension 512, 4 chunks |
| Retrieval | Top-K = 8 at chunk and frame levels |
| Precision | bfloat16 with DeepSpeed ZeRO-3 |
| SFT optimization | AdamW, learning rate 1e-4, 200 steps |
| Sequence length | 5,120 |
| Training hardware | 8 x NVIDIA A100 80 GB |

Full commands and configuration files will be added with the code release.

## Release Status

- [ ] Training and inference code
- [ ] Evaluation scripts
- [ ] Model checkpoints
- [ ] nuScenes-10K generation tools and annotations
- [ ] Paper and supplementary material links


## Citation

If you find ConsiSpace useful, please cite:

```bibtex
@article{huang2026consispace,
  title   = {ConsiSpace: Learning Geometric Consistency Matters for Video Spatial Reasoning},
  author  = {Huang, Ting and Zhang, Zhenyu and Huang, Wenyuan and Yang, Jian and Tang, Hao},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Contact

For questions, please contact [Zhenyu Zhang](mailto:zhangjesse@foxmail.com). Author homepages: [Ting Huang](https://believeht029.github.io/), [Zhenyu Zhang](https://jessezhang92.github.io/), and [Hao Tang](https://ha0tang.github.io/).
