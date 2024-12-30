# SDR-GNN: Spectral Domain Reconstruction Graph Neural Network for incomplete multimodal learning in conversational emotion recognition
## 运行环境
+ Python 3.8.18
+ CUDA 11.6
+ torch 1.12.0
+ torch-geometric 2.4.0

(详情请见requirements.txt）

## 使用说明
以下是本研究使用到的数据集

[IEMOCAP](https://sail.usc.edu/iemocap/index.html)、[CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)、[CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)

我们还提供代码所用到的[数据集特征](https://pan.baidu.com/s/1mts1_R8Lq2SZ-eUQChDCfg?pwd=sdr1) 

请将下载好的数据集特征放到`features/`目录下

进入`path.py`文件，修改路径以适配您的运行环境

### 运行SDR-GNN（IEMOCAPSix）
```python
python -u train.py --epoch=100 --lr=0.001 --hidden=200 --mask-type='constant-0.4' --windowp=3 --windowf=3 --base-model='GRU' --loss-recon --dataset='IEMOCAPSix' --audio-feature='wav2vec-large-c-UTT' --text-feature='deberta-large-4-UTT' --video-feature='manet_UTT' --seed=55
```

### 可能出现的问题：
**feature not exist**

该错误通常是由于数据特征文件路径不正确或文件缺失导致的。请确认特征文件是否已正确下载并放置在 `features/` 目录下。

**ValueError**

该错误一般是由于输入特征的形状不符合要求，可能与 `torch` 或 `torch-geometric` 版本不同有关。建议确保环境中安装的版本一致，或者手动调整数据的形状以适配。

### 结果保存：
运行结果会保存在`result/`下，不同的硬件和运行环境可能导致结果有所不同，建议尝试不同的超参数设置或随机数种子，以获得最佳效果。

我们建议超参数设置在`--hidden=200`、`--windowp=3`、`--windowf=3`、`--loss-recon=0.5`附近（IEMOCAPFour）。

（详情请见[SDR-GNN](https://arxiv.org/pdf/2411.19822?), p13, Table 6）

## 论文相关
论文链接：[SDR-GNN: Spectral Domain Reconstruction Graph Neural Network for incomplete multimodal learning in conversational emotion recognition](https://arxiv.org/pdf/2411.19822?)

若您觉得我们的工作对您的研究有所帮助，希望您能引用我们的论文，感谢您的支持！

```plain
@article{fu2024sdr,
  title={SDR-GNN: Spectral Domain Reconstruction Graph Neural Network for incomplete multimodal learning in conversational emotion recognition},
  author={Fu, Fangze and Ai, Wei and Yang, Fan and Shou, Yuntao and Meng, Tao and Li, Keqin},
  journal={Knowledge-Based Systems},
  pages={112825},
  year={2024},
  publisher={Elsevier}
}
```

