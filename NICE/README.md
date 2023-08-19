# [NICE : Non-linear Independent Components Estimation(2014)](https://arxiv.org/pdf/1410.8516.pdf)       

+ Deep Learning으로 구현한 최초의 Normalizing Flow 모델
----------------------------------------------------------------------------------------------------        

## Motivation     

+ Normalizing FLow는 NICE에서 처음 제안한 방법이 아니다.    
  - 이전부터 존재했던 Generative 방법 중 하나였다.     
  - 하지만 Normalizing Flow를 deep learning model로 구현하기에는 여러 어려움이 존재했다.   

+ 기존 방법의 문제점    
  - 첫 번째, 대부분의 딥러닝 모델에서 사용하는 함수는 역함수가 존재하지 않는다는 것이다. $\rightarrow$ 즉, Invertible function이 아니라는 것이다.       
  
  - 두 번째, 같은 이유로 대부분의 딥러닝 모델 함수들은 Jacobian Determinant를 구하기 어렵다는 문제점이 있었다.    

----------------------------------------------------------------------------------------------------        

## Contribution    

+ Non-linear transformation을 통한 independent components estimation    
  > 저자는 비선형적인 변환을 사용하여 high-dimension data의 독립 성분을 효과적으로 추정하는 방법을 제안한다.     
  >            
  > 이러한 비선형 변환은 데이터를 낮은 차원(latent space)으로 변환하고, 그 후에 독립 성분을 추정한다.     
  >            
  > 이는 선형 변환만을 고려하는 기존의 방법과는 달리 더 복잡한 데이터 구조를 잘 모델링할 수 있는 장점을 가진다.    

+ Invertible network      
  > 논문에서 제안하는 변환은 invertible한 구조를 가진다.    
  >            
  > 이는 추정된 독립 성분을 다시 원래의 고차원 공간으로 복원할 수 있는 방법을 제공한다.   
  >            
  > 이러한 invertible한 구조는 실제 응용으로써 유용하며, 원본 데이터의 복원 또는 새로운 데이터 생성과 같은 task에 활용할 수 있다.    

+ 이러한 Contributions를 통해 non-linear independent components estimation 문제에 새로운 관점을 제시한다.    

----------------------------------------------------------------------------------------------------        

## Abstract    

+ 논문에서는 complex high-dimensional densities를 modeling하는 **NICE**를 소개한다.   
  - 주어진 데이터들에 대해서, 적당한 latent space로 보내는 non-linear deterministic transformation을 학습한다.   

+ 이러한 transformation을 jacobian determinant 그리고 Jacobian의 역이 유일하게 하면서 complex non-linear transformation을 갖도록 구성한다.    

+ training criterion이 **exact log-likelihood**여서 굉장히 **tractable** 한 것이 장점이다.  

----------------------------------------------------------------------------------------------------        

## Introduction

+ Deep learning approaches은 가장 중요한 변화 요인을 포착하는 데이터 representation 학습에 의존한다.   
  - 그렇다면, "*What is a good representation?*"       
  - 바로 data에 대한 distribution을 모델링하기 쉬운 것을 의미한다.   

+ 논문에서는 data를 새로운 space로 보내는 $h \ = \ f(x)$를 학습하여 prior distribution을 추정하는 방법을 제시한다.     
  - 이 때, prior distribution은 각 dimension이 독립이라고 가정하여 factorize 할 수 있는 형태여야 한다.   
  - 이를 토대로 아래의 식을 만들 수 있다.   
  - $$p_H(h) \ = \ \prod_{d} p_{h_{d}}(h_d)$$    

+ data spcae $\rightarrow$ new space 로 옮기는 형태라 단순히 변수만 치환해주는 것이 아니라 치환했을 때 space의 단위 또한 달라지기 때문에, 이로 인해 발생하는 변화도 고려해야 한다.    
  - Notation : new PDF - $p_H$, data PDF - $p_X$    
  - $$p_X(x) \ = \ p_H(f(x)) \lvert \det \frac{\partial f(x)}{\partial x} \rvert$$    
  - 위 식은, pdf를 전 구간에서 적분한 값이 1이 되는 것을 이용하여 유도한 것이다.   

+ 중요한 것은 jacobian 행렬식과 inverse 또한 간단해야한다.     
  > **data를 latent space로 보내는 transfomation을 훈련한 후 이것의 역을 적용하여 latent space에서 하나의 latent를 뽑아와서 data를 만들어 낸다.**    
  >       
  > 사실 이것이 궁극적인 목표이다.     
  >       
  > 그렇기 때문에 이 transformation의 inverse를 취하는 것 또한 쉬워야 한다.    

+ 위를 만족시켜줄 수 있도록 input $x$를 split 하여 $x_1$, $x_2$로 나누고 각각에 대해 다르게 transformation을 적용한다,   
  - $$y_1 \ = \ x_1$$      
  - $$y_2 \ = \ x_2 \ + \ m(x_1)$$   
  - 이 때, $m$은 어떠한 complex function(논문 실험 시 **ReLU MLP** 사용)이다. 이 식에서 jacobian determinant는 무조건 1이고, inverse 또한 유일하게 존재한다.   
  - inverse 는 아래와 같다.    
  - $$x_1 \ = \ y_1$$      
  - $$x_2 \ = \ y_2 \ - \ m(y_1)$$   

----------------------------------------------------------------------------------------------------        

## Learning Bijective Transformations of Continuous Probability    

+ Log-likelihood based Generative model에서 우리의 목표는 $\log p(x)$를 maximize하는 것이다.   
  - $p_X(x)$를 real data의 PDF라 생각했을 때, 우리의 목적은 $\log p_X(x)$를 maximize하는게 된다.    

+ 앞서 pdf에 대해 change of variable을 적용할 때 뒤에 jacobian determinant가 붙는 것을 보았다. 이를 통해 $\log p_X(x)$에 대한 식을 다시 써보자.    

$$\log(p_X(x)) \ = \ \log(p_H(f(x))) \ + \ \log \lvert \det \frac{\partial f(x)}{\partial x} \rvert$$   

+ space를 옮겼을 때 $p_H(h)$를 factorized distribution이라고 칭했기 때문에, $\log$를 통해 sum 형태로 표현할 수 있다.   

$$\log(p_X(x)) \ = \ \sum_{d=1}^{D} \log(p_{H_d}(f_{d}(x))) \ + \ \log \lvert \det \frac{\partial f(x)}{\partial x} \rvert$$    

+ 좀 직관적으로 말하자면 $x$는 data들이고, $f$는 MLP(ReLU MLP)라고 생각하면 된다.    
  > MLP의 output에 대해 space의 domain을 바꿔주기 위해 jacobian determinant가 붙는다.    

+ $p_H$를 우리가 접근하기 쉬운, 즉, log probability를 쉽게 뽑아낼 수 있는 **Gaussian distribution이나 logistic distribution**으로 두고 $\log p_X(x)$를 구하기 쉽게 한 다음 이 log-likelihood를 최대화 하는 방향으로 학습하게 된다면, 궁극적으로 목적에 맞는 log-likelihood를 높여줄 수 있다.     
  - 이렇게 훈련을 하게 된다면 $f(\cdot)$는 주어진 data를 우리가 의도한 간단한 distribution으로 축소할 수 있다.   

+ 접근이 쉬운 distributio에서 latent를 뽑아온 다음, 이 $f$의 inverse를 취해 latent vector를 널어주면 실제 이미지 또한 얻어올 수 있다.   
  - 이것이 **Flow-based Generative Model**의 방법이다.    

----------------------------------------------------------------------------------------------------         

## Architecture    

### 1. Triangular Matrix    

$$\left\[\begin{matrix}
v & 0 & 0 & 0 \\
v & v & 0 & 0 \\
v & v & v & 0 \\
v & v & v & v \\
\end{matrix} 
\right\]$$     

+ Jacobian을 Triangular Matrix 형태로 만드는 것이다.   
  - 우선 Matrix가 두 개의 삼각형으로 이루어진 형태이다.   
  - 한쪽은 실수값을, 한 쪽은 모두 0을 갖는다.   

+ 이러한 Triangular Matrix는 Determinant를 구하기 쉽다는 특징이 있다.    
  > 왜내하면, 대각 성분들의 곱으로 표현되기 때문이다.    

+ 따라서 함수 $f$의 jacobian이 Triangular Matrix가 되도록 구현하면 Jacobian의 Determinent를 쉽게 구할 수 있다.    

### 2. Coupling Layer    

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/d4728b4f-e871-4537-b0f3-54d9c8681089' height='500'></p>  

+ Jacobian이 triangular matrix가 되도록 구성한다고 하였다. 이것을 만족하는 구조로 Coupling layer를 사용한다.    

+ Coupling Layer란 Input을 둘로 split한 Matrix이다.     

$$y = \begin{bmatrix} y_1 \\ 
y_2 \end{bmatrix}$$     

+ 그리고 나서 $y_2$는 $y_1$과 적절한 함수를 사용하여 구성해준다. 수식으로 다음과 같이 표현할 수 있다.   

#### 2-1. General Coupling Layer    
$$y_{I_1} \ = \ x_{I_1}$$     

$$y_{I_2} \ = \ g(x_{I_2} ; m(x_{I_1}))$$      

+ 일반적인 Coupling Layer의 형태이다. 이렇게 표현되는 $y$의 jacobian은 다음과 같다.    
  > flow의 inverse에 해당하는 것을 통해 sampling을 진행하게 된다. 그래서 단순히 g를 더하기로 정의함.   

$$\frac{\partial y}{\partial x} \ = \ \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} \\ 
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} 
\end{bmatrix} \ = \ \begin{bmatrix} \frac{\partial}{\partial x_1}x_1 & \frac{\partial}{\partial x_2}x_1 \\ 
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} 
\end{bmatrix} \ = \ \begin{bmatrix} I & 0 \\ 
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} 
\end{bmatrix}$$     

+ 따라서, 결과적으로 구하고자 하는 $\det (J)$는 다음과 같다. (**Motivation 두 번째 문제 해결**)      
  >우측 하단에 해당하는 미분만 구해주면 되는 아주 쉬운 작업이 되었다.    

$$\det \frac{\partial y}{\partial x} \ = \ \det \frac{\partial y_{I_2}}{\partial x_{I_2}}$$      

    
#### 2-2. Additive Coupling Layer    

+ 이제 첫 번째 문제점이었던 **Invertible function** 조건을 생각해보자.   
  - 위 의 수식에서 함수 $g$를 단순히 **Additive function**(+)으로 사용해보자.   
  - 수식은 다음과 같이 표현된다.    

$$y_{I_2} \ = \ x_{I_2} \ + \ m(x_{I_1})$$     

$$x_{I_2} \ = \ y_{I_2} \ - \ m(y_{I_1})$$     

+ 이렇게 Additive function을 사용하면 역함수도 바로 구할 수 있다.     

+ 또한 함수 $m$에는 invertible이나 Determinant 같은 **제약 조건**이 없다.    
  > 즉, Deep Neural Network 같은 복잡한 함수를 사용할 수 있다는 장점이 있다.    
  >       
  > 논문에서는 $m$을 ReLU MLP로 사용하였다.    

+ 게다가 위 jacobian determinant 식을 보면 $g$를 그저 더해주는 함수로 사용했기 때문에 det 값은 1이 나온다.   
  > $$\det \frac{\partial y_{I_2}}{\partial x_{I_2}} \ = \ 1 \ $$     
  > **volume preserving**의 의미를 가짐   

#### Combining coupling layers    

+ 자, 지금까지의 내용을 정리해보면      

  - 기존 방식은 두 가지의 문제점을 가지고 있었음
    > 역함수가 존재하지 않는다.    
    > jacobian determinant를 구하기 어렵다     

  - 1. Coupling Layer를 사용함으로써 두 번째 문제점인 Jacobian determinant를 간단하게 구할 수 있었다.   
    > $$\det \frac{\partial y}{\partial x} \ = \ \det \frac{\partial y_{I_2}}{\partial x_{I_2}}$$      

  - 2. Additive Layer를 사용, 즉 함수 $g$를 단순 더하기 함수로 사용하면 역함수도 바로 구할 수 있었다.   
    > $$y_{I_2} \ = \ x_{I_2} \ + \ m(x_{I_1})$$     
    >       
    > $$x_{I_2} \ = \ y_{I_2} \ - \ m(y_{I_1})$$     

  - 3. 게다가 함수 $m$은 Invertible이나 Determinant 같은 **제약 조건이 없기** 때문에 **DNN 같은 복잡한 함수를 사용할 수 있다는 장점**을 가진다.  

+ 이렇게 구성된 Coupling Layer는 복잡한 Distribution을 표현하기 위해 **3~4개의 coupling layer를 연속으로 배치**하여 구성한다.     

----------------------------------------------------------------------------------------------------       

## Experiments    

### Log likelihood and generation    

+ 저자는 **MNIST, Toronto Face Dataset(TFD), Street View House Numbers dataset(SVHN), 그리고 CIFAR-10 dataset**에 대해 학습하고 log-likelihood를 측정했다.  
  - 학습하기 전에 dequantized preprocessing을 수행하였다.   
     - dequantized란?   
        > 먼저 데이터 양자화(quantization)는 연속적인 값을 유한한 수의 불연속한 값으로 표현하는 것을 의마한다.     
        >       
        > 이때, 양자화 level은 데이터의 정밀도를 나타낸다.     
        >       
        > 하지만, 양자화된 데이터는 정보의 일부가 손실되거나 왜곡될 수 있다.     
        >       
        > 이러한 왜곡를 최소화하고 원본 데이터와 유사한 형태로 복구하기 위해 dequantization을 사용한다.   
        >       
        > 즉, 양자화된 값을 다시 연속적인 값으로 변환하는 process를 말한다.    

    - $\frac{1}{256}$의 uniform noise를 추가하고 [0 ~ 1]로 rescale를 수행했다.   
    
    - CIFAR-10 dataset에 대해서는, $\frac{1}{256}$의 uniform noise를 추가하고 [-1 ~ 1]로 rescale를 수행했다.   

  - MNIST, SVHN, CIFAT-10 dataset에 대해서는 standard logistic distribution을 prior distribution을 사용하여 학습했고, TFD dataset에 대해서는 standard normal distribution을 사용했다.   

  - Epochs = 1500, Adam with learning rate $10^{-3}$, momentum 0.9, $\beta_{2}$ = 0.01, $\lambda$ = 1, and $\epsilon$ = $10^{-4}$    
    

+ 이렇게 학습한 NF model의 image classification dataset에 대한 log-likelihood를 측정한 결과이다.    

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/cf76c6ba-c920-4c14-93b6-94c05d9d4aa2'></p>     

+ 다음은 각 데이터셋에 대해 생성한 이미지 결과이다.   

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/687bc861-4a1d-4d9a-a310-aa0684f2247a'></p>      

+ MNIST의 경우 제법 그럴 듯하게 생성하는 것을 볼 수 있다.  

+ 반면에 TFD, SVHN, CIFAR-10 등 조금 더 복잡한 이미지에 대해서는 현저하게 형태가 뭉개진 모습을 볼 수 있다.   

+ 아직은 GANs과 같은 기존 방식 대비 저조한 성능을 보이지만, 이러한 모습은 이후 **RealNVP, GLOW** 등의 모델에서 개선된다.  

### Inpainting    

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/3caff31d-73a3-4c41-9bb7-57e7b28b9ea7'></p>       

+ Inpainting이란?    
  > 이미지에 손실된 부분을 재구성하는 task    

+ Observed dimensions $x_O$을 그 값에 고정시키고, hidden dimensions $X_H$에 대한 log-likelihood를 maximize한다.   
  - 이를 위해 **projected gradient ascent**를 사용하며, 입력 값을 원래 값 범위 내에 유지하기 위해 Gaussian Noise를 추가한다.   
  
  - 이 과정에서 각 iteration마다 step size는 $a_i \ = \ \frac{10}{100 + i}$로 설정했다. 이후 stochastic gradient update를 수행하게 된다. 

----------------------------------------------------------------------------------------------------       

## Conclusion    

+ A novel architecture of highly non-linear bijective transforamtion    
  - 논문에서 제안한 주요 아이디어는 data를 factorized 공간으로 mapping하는 매우 비선형적이면서 양뱡향(invertible) 변환을 학습하는 flexible한 아키텍처이다.     
  - 이는 데이터의 복잡한 관계와 구조를 효과적으로 모델링할 수 있도록 도와준다.    

+ Learning for Maximization Log-liklehood     
  - log-likelihood를 직접 maximize하는 framework를 사용하여 모델을 훈련한다.    
  - 이는 학습 데이터에 가장 잘 적합한 변환을 찾는 데에 도움이 되고, 생성된 결과의 품질을 항샹시킨다.    

+ Efficient Ancestral Sampling    
  - NICE는 효율적이고 unbiased ancestral sampling을 특징으로 한다.     
  - 이는 생성된 데이터를 추출하는 과정에서 효율성과 정확성을 보장할 수 있다.   

+ Competitive log-likelihood results    
  - 실험 결과로서 NICE 모델은 log-liklihood 관점에서 경쟁력 있는 performance를 보여줬다.  
  - 이는 모델이 데이터의 분포를 효과적으로 학습하고 생성할 수 있음을 나타낸다고 볼 수 있다.   

+ Various approximation inference possible    
  - 다양한 귀납 원리을 활용하여 훈련할 수 있다.    
  - 이러한 다양성은 모델의 유연성과 확장성을 높여준다.    

+ Scalability through powerful approximate inference    
  - NICE 모델은 더 복잡한 posterior distribution approximation이나 family of prior distribution을 사용하여 더 강력한 추론을 가능하게 한다.    
  - 이는 모델의 활용성과 성능을 확장하는 열쇠이다.     
