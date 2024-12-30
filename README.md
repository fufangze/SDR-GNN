# SDR-GNN: Spectral Domain Reconstruction Graph Neural Network for incomplete multimodal learning in conversational emotion recognition
<font style="color:rgb(31, 35, 40);">An official pytorch implementation for the paper: SDR-GNN: Spectral Domain Reconstruction Graph Neural Network for incomplete multimodal learning in conversational emotion recognition</font>

You can find the README in Chinese version [here](README.zh-CN.md).

## Environment
+ Python 3.8.18
+ CUDA 11.6
+ torch 1.12.0
+ torch-geometric 2.4.0

(For details, see requirements.txt)

## Usage
The following datasets are used in this research:

[IEMOCAP](https://sail.usc.edu/iemocap/index.html), [CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/), [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)

We also provide the [dataset features](https://pan.baidu.com/s/1mts1_R8Lq2SZ-eUQChDCfg?pwd=sdr1) used in the code.

Please place the downloaded dataset features in the `features/` directory.

Open the `path.py` file and modify the paths to fit your environment.

### Running SDR-GNN (IEMOCAPSix)
```python
python -u train.py --epoch=100 --lr=0.001 --hidden=200 --mask-type='constant-0.4' --windowp=3 --windowf=3 --base-model='GRU' --loss-recon --dataset='IEMOCAPSix' --audio-feature='wav2vec-large-c-UTT' --text-feature='deberta-large-4-UTT' --video-feature='manet_UTT' --seed=55
```

### Potential Issues:
**feature not exist**

This error usually occurs due to incorrect file paths or missing feature files. Please confirm that the feature files have been correctly downloaded and placed in the `features/` directory.

**ValueError**

This error typically occurs when the shape of the input features does not match the required format. It may also be related to compatibility issues between the versions of `torch` or `torch-geometric`. It is recommended to ensure that the installed versions are consistent, or manually adjust the shape of the data to fit the requirements.

### Saving Results:
The results will be saved in the `result/` directory. Different hardware and runtime environments may lead to variations in results, so it's recommended to try different hyperparameter settings or random seeds to achieve the best performance.

We recommend setting hyperparameters around `--hidden=200`, `--windowp=3`, `--windowf=3`, `--loss-recon=0.5` (IEMOCAPFour).

(For details, see [SDR-GNN](https://arxiv.org/pdf/2411.19822?), p13, Table 6)

## Paper Information
Paper link: [SDR-GNN: Spectral Domain Reconstruction Graph Neural Network for incomplete multimodal learning in conversational emotion recognition](https://arxiv.org/pdf/2411.19822?)

If you find our work helpful for your research, we kindly request that you cite our paper. Thank you for your support!

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



