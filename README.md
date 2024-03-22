# Remote Sensing Super-resolution Model Collection

The code of paper: [GCRDN: Global Context-Driven Residual Dense Network for Remote Sensing Image Super-Resolution](https://ieeexplore.ieee.org/abstract/document/10115440)

**Train**

```python src/main.py ```

The project also contains serval methods except from gcrdn including rdn, nlsn, rcan, dbpn, edrn, esrt, swinir, transms. The code of gcrdn is presented at src/model/gcrdn/mymodel.py

* RDN : [Residual Dense Network for Image Super-Resolution](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Residual_Dense_Network_CVPR_2018_paper.pdf)

* NLSN : [Image Super-Resolution with Non-Local Sparse Attention](http://openaccess.thecvf.com/content/CVPR2021/papers/Mei_Image_Super-Resolution_With_Non-Local_Sparse_Attention_CVPR_2021_paper.pdf)

* RCAN : [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf)

* DBPN : [Deep Back-Projection Networks For Super-Resolution](https://openaccess.thecvf.com/content_cvpr_2018/papers/Haris_Deep_Back-Projection_Networks_CVPR_2018_paper.pdf)

* EDRN : [Encoder-Decoder Residual Network for Real Super-resolution](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Cheng_Encoder-Decoder_Residual_Network_for_Real_Super-Resolution_CVPRW_2019_paper.pdf)

* ESRT : [Transformer for Single Image Super-Resolution](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Lu_Transformer_for_Single_Image_Super-Resolution_CVPRW_2022_paper.pdf)

* SwinIR : [SwinIR: Image Restoration Using Swin Transformer](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.pdf)

* Transms : [TranSMS: Transformers for Super-Resolution Calibration in Magnetic Particle Imaging]([https://arxiv.org/abs/2309.01377](https://arxiv.org/pdf/2111.02163.pdf))

**Test**

1. Put pre-trained model into 'model'
2. Change the model name in the option.py

```python test.py```

**Cite**

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
```
