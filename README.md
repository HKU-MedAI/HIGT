# HIGT: Hierarchical Interaction Graph-Transformer for Whole Slide Image Analysis

### [Tutorial](https://github.com/HKU-MedAI/HIGT/tree/main/tutorials/tut_HIGT.ipynb) | [Paper](https://arxiv.org/abs/2309.07400)


### Authors

Ziyu Guo, Weiqin Zhao, Shujun Wang, Lequan Yu


## Overview:

![alt text](/pics/image.png)


## Update:

Sorry for the late update. Then this is a tutorial wrote before about the use of **[HIGT](https://github.com/HKU-MedAI/HIGT/tree/main/tutorials/tut_HIGT.ipynb)** and **[MulGT](https://github.com/HKU-MedAI/HIGT/tree/main/tutorials/tut_MulGT.ipynb)** (including data preprocessing and training and testing). 

In order to unify the entire process, the tutorial is implemented based on **[CLAM](https://github.com/mahmoodlab/CLAM)**. And in order to facilitate the deployment of the entire library, please follow the steps below under Ubuntu. And the *core_utils_re.py*, *create_splits_seq_re.py* and *extract_features_fp_re.py* are re-writed version of corresponding python file in CLAM.

### 1.
```
!git clone https://github.com/mahmoodlab/CLAM.git .
```

### 2.
```
!mv /CLAM/* /
```

Finally, sincerely hope this tutorial can help you solve your problems. And if there are any problems when using it, please leave a issue and we will provide feedback as soon as possible.