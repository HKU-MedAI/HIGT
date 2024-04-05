# HIGT: Hierarchical Interaction Graph-Transformer for Whole Slide Image Analysis

### [Tutorial](https://github.com/HKU-MedAI/HIGT/tree/main/tutorials/tut_HIGT.ipynb) | [Paper](https://arxiv.org/abs/2309.07400)


### Authors

Ziyu Guo, Weiqin Zhao, Shujun Wang, Lequan Yu


## Overview:

![alt text](/pics/image.png)


## Update:

We apologize for the delay in updates. Below is a tutorial that encompasses the usage of **[HIGT](https://github.com/HKU-MedAI/HIGT/tree/main/tutorials/tut_HIGT.ipynb)** and **[MulGT](https://github.com/HKU-MedAI/HIGT/tree/main/tutorials/tut_MulGT.ipynb)** , covering aspects from data preprocessing to the training and testing phases. This tutorial is designed with the goal of standardizing the entire process and is implemented on the foundation of **[CLAM](https://github.com/mahmoodlab/CLAM)** for ease of use and deployment.

To ensure a seamless setup, please follow the instructions below for Ubuntu systems. Note that *core_utils_re.py*, *create_splits_seq_re.py*, and *extract_features_fp_re.py* are revised versions of their respective files in CLAM, changed a little bit for more convenience with HIGT.

### 1.
```
!git clone https://github.com/mahmoodlab/CLAM.git .
```

### 2.
```
!mv /CLAM/* /
```

We hope this tutorial aids you in your projects. Should you encounter any issues, please don't hesitate to raise an issue on our GitHub page. Our team is committed to providing prompt and helpful feedback.