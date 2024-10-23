# Remote Sensing Super-resolution Model Collection

The code of paper: [GCRDN: Global Context-Driven Residual Dense Network for Remote Sensing Image Super-Resolution](https://ieeexplore.ieee.org/abstract/document/10115440)

# Usage

**Train**

You could change all the setting in the option.py through form of '--xxx xxx' during training and testing such as:

```python src/main.py --model your_model_name --save your_save_dir_name```

The project also contains serval methods except from gcrdn including rdn, nlsn, rcan, dbpn, edrn, esrt, swinir, transms. The code of gcrdn is presented at src/model/gcrdn/mymodel.py

* RDN : [Residual Dense Network for Image Super-Resolution](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Residual_Dense_Network_CVPR_2018_paper.pdf)

* NLSN : [Image Super-Resolution with Non-Local Sparse Attention](http://openaccess.thecvf.com/content/CVPR2021/papers/Mei_Image_Super-Resolution_With_Non-Local_Sparse_Attention_CVPR_2021_paper.pdf)

* RCAN : [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf)

* DBPN : [Deep Back-Projection Networks For Super-Resolution](https://openaccess.thecvf.com/content_cvpr_2018/papers/Haris_Deep_Back-Projection_Networks_CVPR_2018_paper.pdf)

* EDRN : [Encoder-Decoder Residual Network for Real Super-resolution](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Cheng_Encoder-Decoder_Residual_Network_for_Real_Super-Resolution_CVPRW_2019_paper.pdf)

* ESRT : [Transformer for Single Image Super-Resolution](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Lu_Transformer_for_Single_Image_Super-Resolution_CVPRW_2022_paper.pdf)

* SwinIR : [SwinIR: Image Restoration Using Swin Transformer](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.pdf)

* TranSMS : [TranSMS: Transformers for Super-Resolution Calibration in Magnetic Particle Imaging](https://arxiv.org/pdf/2111.02163.pdf)

**Test**

1. Put pre-trained model into 'pre_train'
2. Change the model name in the option.py or use '--model your_model_name' :

```python src/test.py --model your_model_name --save your_save_dir_name```

My pretrained files on OLI2MSI of all models mentioned above are uploaded which could be gained from [https://pan.baidu.com/s/1Zw8Vww-dLX_sRHYVdtQBww](https://pan.baidu.com/s/1Zw8Vww-dLX_sRHYVdtQBww) code: been

**Dataset**

The experimental datasets, OLI2MSI and Alsat, could be obtained from:

* [OLI2MSI](https://github.com/wjwjww/OLI2MSI)

* [Alsat](https://github.com/achrafdjerida/Alsat-2B)

**Env**

pytorch==1.13.0

cuda==11.7

python==3.10.6

# Cite

Please cite this paper :)
```
@ARTICLE{10115440,
  author={Sui, Jialu and Ma, Xianping and Zhang, Xiaokang and Pun, Man-On},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={GCRDN: Global Context-Driven Residual Dense Network for Remote Sensing Image Superresolution}, 
  year={2023},
  volume={16},
  number={},
  pages={4457-4468},
  doi={10.1109/JSTARS.2023.3273081}}

@article{sui2024denoising,
  title={Denoising Diffusion Probabilistic Model with Adversarial Learning for Remote Sensing Super-Resolution},
  author={Sui, Jialu and Wu, Qianqian and Pun, Man-On},
  journal={Remote Sensing},
  volume={16},
  number={7},
  pages={1219},
  year={2024},
  publisher={MDPI}
}

@article{sui2024adaptive,
  title={Adaptive Semantic-Enhanced Denoising Diffusion Probabilistic Model for Remote Sensing Image Super-Resolution},
  author={Sui, Jialu and Ma, Xianping and Zhang, Xiaokang and Pun, Man-On},
  journal={arXiv preprint arXiv:2403.11078},
  year={2024}
}
```
If you have any questions, be free to contact me!
