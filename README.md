# BRBS! Learning Better Registration to Learn Better Few-Shot Medical Image Segmentation: Authenticity, Diversity, and Robustness

We address the task of few-shot medical image segmentation (MIS) with a novel proposed framework based on the learning registration to learn segmentation (LRLS)
paradigm. To cope with the limitations of lack of diversity, authenticity, and robustness in the existing LRLS frameworks, we propose the Better Registration Better Segmentation (BRBS) framework with three main contributions that are experimentally shown to have substantial practical merit. Without any bells and whistles, our approach achieves a new
state-of-the-art performance in few-shot MIS, and we believe that this novel and effective framework will provide a powerful few-shot benchmark for the field of medical image and efficiently reduce the costs of medical image research.


<p align="center"><img width="100%" src="fig/detil.png" /></p>

## Paper
This repository provides the official tensorflow implementation of PC-Reg-RT in the following papers:

**Learning Better Registration to Learn Better Few-Shot Medical Image Segmentation: Authenticity, Diversity, and Robustness** <br/> 
[Yuting He](http://19951124.academic.site/?lang=en), [Guanyu Yang*](https://cse.seu.edu.cn/2019/0103/c23024a257233/page.htm), Rongjun Ge, Xiaoming Qi, Yaolei Qi, [Shuo Li*](http://www.digitalimaginggroup.ca/members/shuo.php) <br/>
Southeast University <br/>

The paper is under review, and the complete code will be opened after this paper is published.


## Available implementation
- [tensorflow/](https://github.com/YutingHe-list/PC-Reg-RT/tree/main/tensorflow)
- [pytorch/](https://github.com/YutingHe-list/PC-Reg-RT/tree/main/pytorch)

## Citation
If you use PC-Reg-RT for your research, please cite our papers:
```
@ARTICLE{9477084,
  author={He, Yuting and Li, Tiantian and Ge, Rongjun and Yang, Jian and Kong, Youyong and Zhu, Jian and Shu, Huazhong and Yang, Guanyu and Li, Shuo},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Few-shot Learning for Deformable Medical Image Registration with Perception-Correspondence Decoupling and Reverse Teaching}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JBHI.2021.3095409}}
```

## Acknowledgments

This research was supported by the National Key Research and Development Program of China (2017YFC0109202), National Natural Science Foundation under grants (31800825, 31571001, 61828101), Excellence Project Funds of Southeast University and Scientific Research Foundation of Graduate School of Southeast University (YBPY2139). We thank the Big Data Computing Center of Southeast University for providing the facility support on the numerical calculations in this paper.
