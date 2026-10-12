# MoCha-Stereo (MoCha-V2): faster inference fork

> **You are on `fast`.** Research fork of [ZYangChen/MoCha-Stereo](https://github.com/ZYangChen/MoCha-Stereo) (base `78c5889`; `main` mirrors upstream). Branches:
> - [`fast`](https://github.com/fastforks/MoCha-Stereo/tree/fast): plain PyTorch, no build step, no new dependency, exports with `torch.export`.
> - [`kernels`](https://github.com/fastforks/MoCha-Stereo/tree/kernels): `fast` plus custom CUDA / CUTLASS kernels in a separate fp16 path (`Mocha.forward_fast`; prebuilt for sm_80 to sm_120).
> - [`pr/*`](https://github.com/fastforks/MoCha-Stereo/branches): the four small changes proposed upstream, one branch each (`pr/combined` = all four).
>
> Licence, as upstream: code MIT · weights: linked from the upstream README, not re-hosted here.

MoCha-Stereo (released V1 checkpoint, stock code incl. the stock MCCV), 32 iterations, input 1482x994 (Middlebury-H size, padded to /32). RTX 5060 Ti 16 GB, torch 2.14.1+cu130, 2026-10-12. Median of 10 runs after warm-up; memory = torch peak allocated / reserved. Accuracy: mean |disparity difference| to upstream fp32 on 7 ETH3D + 5 Middlebury-H training pairs (px), and ground-truth EPE.

| level | ms | speed-up | peak MB alloc / reserved | accuracy vs upstream fp32 |
|---|---|---|---|---|
| upstream default (fp32) | 3252 | 1.00x | 3763 / 4670 | reference (EPE ETH3D 0.317, Midd-H 0.941) |
| + realistic upstream PRs, no flags (fp32; bitwise equal output) | 3036 | 1.07x | 3761 / 4670 | identical |
| + flags only (`mixed_precision`, channels_last, cudnn.benchmark, TF32) | 1395 | 2.33x | 3420 / 4074 | 0.003 / 0.013 px; EPE 0.318 / 0.938 |
| + realistic upstream PRs (`pr/combined`, with the flags) | 1188 | 2.74x | 2517 / 3386 | 0.003 / 0.013 px; EPE 0.318 / 0.939 |
| fork `fast` (friendly) | 1188 | 2.74x | 2517 / 3386 | identical to the row above |
| fork `kernels` (max) | 487 | 6.68x | 1636 / 2282 | 0.004 / 0.014 px; EPE 0.318 / 0.939 |

**Realistically mergeable upstream: 2.74x of 6.68x**

All friendly changes are small enough to be upstream PRs, so `fast` is the same code as `pr/combined` (rows 4 and 5 are one measurement). Cost of the flags row: start-up 4.0 s, first call 9.0 s (cudnn.benchmark; about 1.1 s without it), new input size 4.3 s with cudnn.benchmark. `kernels`: start-up 4.2 s, first call 0.72 s, new size 0.17 s. KITTI size: 1319 ms upstream, 369 ms `fast`, 174 ms `kernels`.

Realistic upstream PRs (ms saved with fp16 autocast alone, vs 1483 ms for autocast alone; 1306 ms all four together; they do not add up exactly):
- `pr/merged-lstm-gates`: the four LSTM gate convs as one conv, -78 ms (draft `02_PR_merged_lstm_gates.md`)
- `pr/remp-autocast`: REMP refinement under the autocast flag, -73 ms and -376 MB reserved (-558 MB allocated) (`04_PR_remp_autocast.md`)
- `pr/sync-free-lookup`: no host sync in `bilinear_sampler` (the `torch.unique` assert blocks `torch.export`), -12 ms (`01_PR_sync_free_lookup.md`)
- `pr/skip-mask-feat`: upsampling-mask features only on the last iteration, -9 ms (`03_PR_skip_mask_feat.md`)

Fork-only: the `kernels` path ships custom CUDA / CUTLASS kernels, prebuilt binaries and a second inference code path; far too large and too hardware-specific to review as an upstream PR.

Note: measured with the released MoCha-Stereo (V1) checkpoint on the unmodified upstream code. The released V2 checkpoint holds weights for a V2 `CCV` module that is not in the upstream repository, so it cannot be run with the stock MCCV.

---

# MoCha-Stereo 抹茶算法
+ [CVPR2024] The official implementation of "[MoCha-Stereo: Motif Channel Attention Network for Stereo Matching](https://github.com/ZYangChen/MoCha-Stereo/tree/main/MoCha-Stereo)".
+ [arXiv] The extension version of MoCha-Stereo. "Motif Channel Opened in a White-Box: Stereo Matching via Motif Correlation Graph"

<div align="center">
  <img width="80%", src="./mocha-robot1.jpg">
</div>


## V1 Version

<div align="left">
    <a href="https://openaccess.thecvf.com/content/CVPR2024/html/Chen_MoCha-Stereo_Motif_Channel_Attention_Network_for_Stereo_Matching_CVPR_2024_paper.html" target='_blank'><img src="https://img.shields.io/badge/CVPR-2024-9cf?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAjCAMAAAADt7LEAAADAFBMVEVMaXF5mcqXsNZ0lceQq9OXsNaHpM9zlMeHo8+dtNh9nMypvt12lsiFos6mu9xxksZvkcZzlMdlicJxksZqjcRukMUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACreRF2AAAAEXRSTlMA1mzoi0mi9BQ1wiRqVgeN+nWbeCoAAAAJcEhZcwAALiMAAC4jAXilP3YAAAJASURBVDiNnZTbduIwDEW3ZJOEwBTo/P8ntnRCCQHb8jzkUhgIrDV6iRN7W0eXCP7HZFotJKiPtsjh7pDfHq5f68YPq1/tNksEFfmsD/9iVt/e0lNvsaxivw/vuV01zxUqQHWsU5w+RbPnDHjg7bBLgCqmOZt854euguRhtfBAt8ugshfxMedNd3l4fyvdta/FWjKaDwkCVF/FK309haGyH4Lp6J4TAMqyFsjNywzcUsEMlcXLk8sbhYUmpOkz8BYA9DhtVwJk0wSklQGnnjotgaFaTV05U+w0UrZ2pG8DkLUqNFUHKGqQV8OpAMSUJlVvzkxdJX0wU/mVsRN7qsuGbMYYkkWTj5OLAGZmcaSigkyRBMBGZ6vagQl3ppQKNjkMrQPZ9IrPGvF/1uPWd4yxKPpsXCpwqRprGwDr/7EquEiSoSlbCdOfp4hC3Ewtf+Xssol4K+8FooQPB243lbmPLABZIbXHB5QDLcG0dEOrG2XGVxJ0Z6iLcah1kHhNZVmAOQdFLVV5oQDLrS2LDIczULl8Sykg5iCm1XZ5XhYtoXPgki4TqhngmLk1B4RUuAxmWkiuzrlSsLoAtD1DH8OdL8Lp4BQwMzRz2DswN176MIcAp7JR5xVQ2SlrAwzcx92M+1EInJMcj6U67LNw4dKt+kjO/UNLFXHxR+F1k1USfe4AdJsB/znMxdXFE/2ieUhdmfxOIF9zY0FnKAN/nJ1WM5TtHSnNTqsZCjF1j/r2hcn7k2k65wuZq/BTW/knm38BWrgDGcRH1DMAAAAASUVORK5CYII="/></a>&nbsp;
    <a href="https://arxiv.org/pdf/2404.06842.pdf" target='_blank'><img src="https://img.shields.io/badge/Paper-PDF-f5cac3?logo=adobeacrobatreader&logoColor=red"/></a>&nbsp;
    <a href="https://openaccess.thecvf.com/content/CVPR2024/supplemental/Chen_MoCha-Stereo_Motif_Channel_CVPR_2024_supplemental.pdf" target='_blank'><img src="https://img.shields.io/badge/Supp.-PDF-f5cac3?logo=adobeacrobatreader&logoColor=red"/></a>&nbsp; 
	<a href="https://kns.cnki.net/kcms2/article/abstract?v=VkQzIsHyPdiXOZ6uXUyfivU9sw0L6aCigQddB8XY3kQv2xxDD_PZgfE930ZrL792l6Ja8IZ4Q_rHF8P3ZJmixyHK5a8qnFYkwDoNMPsQqWXyV9Onp09yYpnB13Ge2IcnPc0ZBSK2p02CFOWrMTXb3KGkB7IK42dsaQcZl4PJP1pd7ZqVhcObiu58-VpKHCNl&uniplatform=NZKPT&language=CHS" target='_blank'><img src="https://img.shields.io/badge/中文版-PDF-f5cac3?logo=adobeacrobatreader&logoColor=red"/></a>&nbsp;
    <a href="https://paperswithcode.com/sota/stereo-disparity-estimation-on-kitti-2015?p=mocha-stereo-motif-channel-attention-network" target='_blank'><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mocha-stereo-motif-channel-attention-network/stereo-disparity-estimation-on-kitti-2015" /></a>
	<!--<a href="https://paperswithcode.com/sota/stereo-depth-estimation-on-kitti-2015?p=mocha-stereo-motif-channel-attention-network" target='_blank'><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mocha-stereo-motif-channel-attention-network/stereo-depth-estimation-on-kitti-2015" /></a>-->
	
</div>

https://github.com/ZYangChen/MoCha-Stereo/assets/108012397/2ed414fe-d182-499b-895c-b5375ef51425

> MoCha-Stereo: Motif Channel Attention Network for Stereo Matching <br>
> [Ziyang Chen](https://scholar.google.com/citations?user=t64KgqAAAAAJ&hl=en&oi=sra)†, [Wei Long](https://scholar.google.com/citations?user=CsVTBJoAAAAJ&hl=en)†, [He Yao](https://scholar.google.com/citations?user=c0qjMAMAAAAJ&hl=en)†, [Yongjun Zhang](http://cs.gzu.edu.cn/2021/1210/c17588a163831/page.htm)✱,[Bingshu Wang](https://teacher.nwpu.edu.cn/wangbingshu.html), [Yongbin Qin](http://cs.gzu.edu.cn/2021/1210/c17588a163794/page.htm), [Jia Wu](https://faculty.csu.edu.cn/jiawu/zh_CN/index.htm) <br>
> CVPR 2024 <br>
> Contact us: ziyangchen2000@gmail.com; zyj6667@126.com✱



```bibtex
@inproceedings{chen2024mocha,
  title={MoCha-Stereo: Motif Channel Attention Network for Stereo Matching},
  author={Chen, Ziyang and Long, Wei and Yao, He and Zhang, Yongjun and Wang, Bingshu and Qin, Yongbin and Wu, Jia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={27768--27777},
  year={2024}
}
```

## V2 Version

<a href="https://chenziyang.cn/project_page/mocha_project/index.html"><img src="https://img.shields.io/badge/Project-Page-blue.svg"></a>
<a href="https://arxiv.org/abs/2411.12426" target='_blank'><img src="https://img.shields.io/badge/Paper-PDF-f5cac3?logo=adobeacrobatreader&logoColor=red"/></a>&nbsp;
<a href="https://kns.cnki.net/kcms2/article/abstract?v=VkQzIsHyPdiXOZ6uXUyfivU9sw0L6aCigQddB8XY3kQv2xxDD_PZgfE930ZrL792l6Ja8IZ4Q_rHF8P3ZJmixyHK5a8qnFYkwDoNMPsQqWXyV9Onp09yYpnB13Ge2IcnPc0ZBSK2p02CFOWrMTXb3KGkB7IK42dsaQcZl4PJP1pd7ZqVhcObiu58-VpKHCNl&uniplatform=NZKPT&language=CHS" target='_blank'><img src="https://img.shields.io/badge/中文版-PDF-f5cac3?logo=adobeacrobatreader&logoColor=red"/></a>&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mocha-stereo-motif-channel-attention-network/stereo-disparity-estimation-on-middlebury)](https://paperswithcode.com/sota/stereo-disparity-estimation-on-middlebury?p=mocha-stereo-motif-channel-attention-network)

<div align="center">
  <img width="80%", src="./mocha-robot2.png">
</div>

> Motif Channel Opened in a White-Box: Stereo Matching via Motif Correlation Graph <br>
> [Ziyang Chen](https://scholar.google.com/citations?user=t64KgqAAAAAJ&hl=en&oi=sra), [Yongjun Zhang](http://cs.gzu.edu.cn/2021/1210/c17588a163831/page.htm)✱,[Wenting Li](https://www.gzcc.edu.cn/jsjyxxgcxy/contents/3205/3569.html), [Bingshu Wang](https://teacher.nwpu.edu.cn/wangbingshu.html), [Yong Zhao](https://www.ece.pku.edu.cn/info/1045/2131.htm), [C. L. Philip Chen](https://www.ieeeiciea.org/2023/Prof.PhilipChen.html) <br>
> Arxiv Report <br>
> Contact us: ziyangchen2000@gmail.com; zyj6667@126.com✱

```bibtex
@article{chen2024motif,
  title={Motif Channel Opened in a White-Box: Stereo Matching via Motif Correlation Graph},
  author={Chen, Ziyang and Zhang, Yongjun and Li, Wenting and Wang, Bingshu and Zhao, Yong and Chen, CL},
  journal={arXiv preprint arXiv:2411.12426},
  year={2024}
}
```

## FAQ
**Q1: Weight for "tf_efficientnetv2_l"?** (Please refer to issue [#6 "关于tf_efficientnetv2_l检查点的问题"](https://github.com/ZYangChen/MoCha-Stereo/issues/6), [#8 "预训练权重"](https://github.com/ZYangChen/MoCha-Stereo/issues/8), and [#9 "code error"](https://github.com/ZYangChen/MoCha-Stereo/issues/9). )

A1: You can download it [here](https://github.com/ZYangChen/MoCha-Stereo/releases/download/tf_efficientnetv2_l-d664b728/tf_efficientnetv2_l-d664b728.pth). Moreover, the weight pretrained on the Scene Flow dataset is available [here](https://github.com/ZYangChen/MoCha-Stereo/releases/tag/MoCha-Stereo-V1).

**Q2: How to visualize the disparity map?** (Please refer to issue [#15 "询问可视化问题"](https://github.com/ZYangChen/MoCha-Stereo/issues/15), and [#10 "请问这个项目如何可视化推理的结果呢"](https://github.com/ZYangChen/MoCha-Stereo/issues/10))

A2: You can accomplish this using "[demo.py](https://github.com/ZYangChen/MoCha-Stereo/blob/main/MoCha-Stereo/demo.py)". 

```
python demo.py --restore_ckpt ./model/mocha-stereo.pth -l ./datasets/Middlebury/MiddEval3/trainingF/*/im0.png -r ./datasets/Middlebury/MiddEval3/trainingF/*/im1.png --output_directory ./your/path
```

The libary "matplotlib"  is required for visualizing the disparity map.


## Acknowledgements
<ul>
<li>This project borrows the code from <strong><a href="https://github.com/gangweiX/IGEV">IGEV</a></strong>, <a href="https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network">DLNR</a>, <a href="https://github.com/princeton-vl/RAFT-Stereo">RAFT-Stereo</a>, <a href="https://github.com/xy-guo/GwcNet">GwcNet</a>. We thank the original authors for their excellent works!</li>
</ul>

