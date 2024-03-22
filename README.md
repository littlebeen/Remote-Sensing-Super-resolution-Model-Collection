# Remote Sencing Super-resolution Model Collection

The code of paper: [GCRDN: Global Context-Driven Residual Dense Network for Remote Sensing Image Super-Resolution](https://ieeexplore.ieee.org/abstract/document/10115440)

**Train**

```python src/main.py ```

**Train**

The project also contains serval methods except from gcrdn including rdn, nlsn, rcan, dbpn, edrn, esrt, swinir, transms and rrdb.

The code of gcrdn is presented at src/model/gcrdn/mymodel.py

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
