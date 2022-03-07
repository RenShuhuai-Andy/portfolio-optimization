# CS410: Portfolio Optimization

Assignment for PKU Network Information Architecture (网络信息体系结构) 2021 spring.

Given the TOP50 stocks (A-share) with heavy public offerings in the first quarter of 2021, determine the efficient frontier and optimal portfolio.

## Introduction

![introduction](figures/introduction.png)

## Prepare environment
```
conda create -n cs410 python=3.6
conda activate cs410
conda install pytorch torchvision cudatoolkit=10.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
pip install -r requirements.txt 
```

## Optimization Model

<details>
<summary>1. Mean-Variance Optimization</summary>

- [Mean-Variance.py](Mean-Variance.py)

![mvo](figures/mvo.png)
</details>

<details>
<summary>2. Capital Asset Pricing Model</summary>

- [CAPM.py](CAPM.py)

![capm](figures/capm.png)
</details>

<details>
<summary>3. Neural Network</summary>

- [nn.ipynb](nn.ipynb) & [NN.py](NN.py)

![nn1](figures/nn1.png)
![nn2](figures/nn2.png)
![nn3](figures/nn3.png)
![nn3](figures/nn4.png)
</details>

## Evaluation

- [evaluation.py](utils/evaluation.py)

## Visualization

- [visualization.ipynb](utils/visualization.ipynb)

## Results

![results1](figures/results1.png)
![results2](figures/results2.png)
![results3](figures/results3.png)
![results4](figures/results4.png)
![results5](figures/results5.png)
![results6](figures/results6.png)