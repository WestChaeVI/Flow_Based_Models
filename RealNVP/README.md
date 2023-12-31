# [Density estimation using Real NVP(2016)](https://arxiv.org/pdf/1605.08803.pdf)        

+ Real NVP (Real-valued Non-Volume Preserving tranformation)     
  - RealNVP는 NICE 모델을 일부 개선한 모델      
-------------------------------------------------------------          
## Motivation    

+ Volume-Preserving transformation만 가능했던 NICE    

  - 기존의 NICE에서는 transformation을 적용할 때 단순 더하기로 진행했기 때문에 volume을 보존하는 성질을 지녔다.    
  - 이는 입력 공간에서 유효한 변환을 적용하는 것이 제약되는 문제를 의미한다.     
  - 이로 인해 더 다양한 데이터 분포의 모델링이 어렵고, 특히 많은 데이터셋에 대해 복잡한 관계를 표현하기 어려웠다.   

+ 이러한 이슈 때문에 $x \rightarrow z$로 가는 방향 기준 마지막에 scaling layer를 구성했다.    

+ 논문에서는 단순 더하기가 아닌 volume을 보존하지 않는 transformation을 차용하였다.    

-------------------------------------------------------------          

## Introduction    

+ NICE는 volume preserving transformation (addition)을 사용했다.     
  - 하지만 우리가 관심있는 데이터들은 고차원으로 구성되는 경우가 많다.     
  - 이는 modeling을 할 때, complexity를 충분히 파악할 수 있게끔 modeling해야 함을 의미한다.    
  > NICE는 volume preserving transformation으로 우리가 원하는 직관적이고 간단한 latent space로 보내게 된다.    
  > 이로 인해, 충분히 complexity를 파악하지 못하게 된다.       

+ 논문에서는 Non-Volume Preserving transformation(NVP) 적용하는 모델을 제안한다.   

-------------------------------------------------------------          

## Model Definition    

+ 한 줄 요약 : **좀 더 flexible한 architecture, 더 강력한 bijective functions 제안.**      

### Change of Variable Formula     

flow based model에서 계속 나오는 내용    

+ Given an observed data variable $x \in X$, a simple prior probability distribution $p_z$ on a latent variable $z \in Z$, and a bijection $f : X \rightarrow Z$ (with $g = f^{-1}$), the change of variable formula defines a model distribution on $X$ by     

$$p_X (x) \ = \ p_Z (f(x)) \lvert \det ( \frac{\partial f(x)}{\partial x^T} ) \rvert$$    

$$\log ( p_X (x) ) \ = \ \log ( p_Z (f(x)) ) \ + \ \log ( \lvert \det ( \frac{\partial f(x)}{\partial x^T} )  \rvert )$$     

where $\frac{\partial f(x)}{\partial x^T}$ is the Jacobian of $f$ at $x$.      

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/f5b5b4e2-d579-4766-bbec-be4e429710d3' width=700></p>    

+ 위에서 설명했던 고차원 데이터를 우리가 원하는 직관적이고 간단한 latent space로 보내는 과정에 대한 Figure이다.    
+ data space는 어떠한 manifold를 형성하는 고차원 데이터 공간이고, 이것을 직관적이고 간단한 고차원 latent space로 보내줄 것이다.    
  > 간단한 고차원 latent space란 NICE에서도 언급했듯이 dimension들이 서로 독립인 직관적인 latent distribution이다. (i.e. Gaussian, Logistic distribution)    

+ Inference 단계에서 $x$를 $z$로 보내줄 것이고, Generation 단계에선 $z$를 $x$로 보내준다.   
  - Inference(train)의 함수를 기반으로 한 inverse 함수를 통해 $z$를 $x$로 보내줄 것이기 때문에 **bijective** 해야한다.      
  - 또한, Jacobian Matrix는 Triangular matrix    

### Coupling Layer      

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/82c50440-0938-4ae1-8efa-06e99ed5effd' width=700></p>      


+ 당연하게 NICE와 마찬가지로 위에 나온 내용들을 잘 만족하도록 구조를 짜야한다.   
  - Coupling layer 형태의 layer를 통해 역함수를 구하기 쉬우며, Jacobian이 triangular matrix 형태를 띄도록 했다.   

+ 특이한 점이 있다면 **scale factor $s$를 통해서 volume을 유지하지 않도록 했다.**     

$$y_{1:d} \ = \ x_{1:d}$$   

$$y_{d+1:D} \ = \ x_{d+1:D} \ \odot \ exp(s(x_{1:d})) \ + \ t(x_{1:d})$$    

+ **수식으로 보는 NICE와의 차이점**       
  - NICE에서는 coupling layer를 사용하고 내부의 $g$ function으로 additive function을 사용했다.    
  - 덕분에 Inversion이 가능하고 Jacobian Determinant를 구하기 쉬워 Normalizing Flow를 구현할 수 있었다.   
  - 하지만 내부 함수 $g$를 **단순하게 더하기(+)를 사용했기에 복잡한 데이터를 표현하기 어렵다는 한계점**이 존재했다.   
  - 이를 개선하기 위해 RealNVP에서는 내부함수 $g$를 **Affine Transformation**으로 구성한다. 
    > 위 식 중 $exp$, $s$, $t$ 로 이루어진 식이 Affine transformation 식이다.      
    >         
    > scale factor에 exp가 붙었는데 이는 Jacobian determinant 구할 때 $\log$를 상쇄시켜주기 위함이다.    
    >      
    > 그리고 $s$와 $t$ factor는 complex network (=ResNet)을 통해 구하게 된다.    



### Properties   

+ Triangular Matrix의 Determinant는 대각 성분의 곱으로 표현된다. 따라서 Affine Transform으로 표현되는 $y$의 Jacobian을 구해보면 다음과 같다.    

$$\frac{\partial y}{\partial x^T} = \begin{bmatrix}
\mathbb{I}\_d & 0 \\
\frac{\partial y_{d+1:D}}{\partial x_{1:d}^T} & \text{diag}(\exp\[s(x_{1:d})\])
\end{bmatrix}$$     

+ 따라서 $\det(J)$는 다음과 같다.    

$$exp \left\[ \sum_{j} s(x_{1:d})_j\right\]$$     

+ Jacobian의 Determinant가 NICE 논문 처럼 아주 간단하게 구해지는 모습을 볼 수 있다.   
  - 게다가 Jacobian의 Determinant를 구할 때 $s$, $t$의 Jacobian을 구하지 않아도 된다.    
  - 따라서, **$s$, $t$는 복잡한 함수를 사용할 수 있다. 즉, DNN으로 표현할 수 있다는 말이 된다.**    

+ Jacobian Determinant를 확인했으니 Inverse또한 가능한지 확인해 봐야한다.      
  - Affine Coupling Layer의 Inversion도 아래와 같이 쉽게 표현된다.   

$$\begin{cases}
    y_{1:d} \ = \ x_{1:d} \\
    y_{d+1:D} \ = \ x_{d+1:D} \odot exp(s(x_{1Ld})) \ + \ t(x_{1:d})
\end{cases}$$      

$$\Leftrightarrow \begin{cases}
    x_{1:d} \ = \ y_{1:d} \\
    x_{d+1:D} \ = \ (y_{d+1:D} \ - \ t(y_{1:d})) \odot exp(s(y_{1Ld}))
\end{cases}$$       


### Masked Convolution     

+ NICE에서는 Coupling Layer의 Input을 단순히 절반으로 split했었다.    
  > 이렇게 되면 나눈 Input의 절반은 변하지 않고 그대로라는 문제가 있다.    

+ 이에 RealNVP 모델에서는 Coupling Layer의 Input Spluc을 위해 Masked Convolution을 사용한다.   
  > 이렇게 되면 다양한 패턴으로 input을 나눠 Coupling Layer로 구성해줄 수 있다.   

+ Masked Convolution의 수식은 다음과 같다.   

$$y \ = \ b \odot x \ + \ (1-b) \odot ( x  \odot exp(s(b \odot x)) \ + \ t(b \odot x) )$$    

+ 이때 Spatial Checkerboard Pattern과 Channel Wise Masking을 적용한다. 따라서 Input은 아래 그림과 같이 분리되게 된다.   

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/8342c02e-afb4-4557-958e-bacba057842f' width=700></p>      

### Combining coupling layers    

+ 지금까지 쭉 바꿔왔는데 그래도 남아있는 문제는 Coupling Layer 절반의 $x$는 Transform 없이 그대로 남아 있다는 것이다.  

+ 이를 해결하기 이해 아래 그림과 같이 Alternating Pattern으로 Coupling Layer를 구성해준다.   

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/d1e99c6e-b470-4aef-bc92-16c035c83995' width=700></p>      


### Multi-scale arichitecture   

+ 이제 각 한층의 형식을 봤으니 Mult-scale을 고려할 수 있는 아키텍처를 생각해보자.   
  - 알다시피 이 flow based model을 구성하는 transformation들은 invertible 해야한다.   
  - 그렇기 때문에 어떤 trick 없이 그저 layer만 쌓는다면 동일한 shape에 대해서만 고려할 수 밖에 없다.    

+ 하지만 Multi-scale에 대해 고려혀줄 수 있게 논문에서는 **Squeezing**이라는 개념을 도입한다.   

+ Squeezing operation이란?      
  > $s$ x $s$ x $c$ 형태의 tensor를  $\frac{s}{c}$ x $\frac{s}{c}$ x $4c$ 로 바꿔주는 것이다.   
  >       
  > 이 방법으로 shape을 변환할 수 있고, 이의 inverse operation 또한 간단하다.    


### Batch Normalization    

+ 위 구조를 토대로 $s$, $t$ factor에 대해 BN과 WN을 적용했다. 뿐만 아니라 Coupling layer의 output 에도 BN을 적용했다.   

+ BN은 input의 입장에서 보면 Rescaling function을 적용한 것과 같다.

$$ x \rightarrow \frac{x - \tilde{\mu}}{\sqrt{\tilde{\sigma^2} + \epsilon}}$$   

+ 이에 따른 Jacobian determinant는 다음곽 같이 나오게 된다.    

$$( \prod_{i} (\tilde{\sigma}_{i}^2 + \epsilon ) )^{-\frac{1}{2}}$$

-------------------------------------------------------------          

## Experiments    

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/a36ebc26-af70-495f-b03e-a796cd2b593c'></p>      

+ 평가지표 : **Bits/dim** (Bits per Dimension)   
  > bits per dimension은 생성된 데이터의 각 차원을 인코딩하는 데 필요한 비트 수를 의미.     
  >      
  > 이 값이 낮을수록 모델이 더 좋은 품질의 데이터를 생성한다는 것을 나타낸다.      
  >       
  > 따라서 낮은 Bits/dim 값은 생성된 데이터가 실제 데이터 분포에 더 가깝게 모사되었다는 것을 나타내며, 이는 생성 모델의 성능이 높다는 것을 의미하게 된다.      

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/40abeab6-3bcb-463e-bebc-24d0caa30c03' height='600'></p>    

+ scaling function $s(\cdot)$ : $Tanh(\cdot)$   
+ translation function $t(\cdot)$ : affine output   
+ prior $p_Z$ : isotropic unit norm Gaussian    
+ batch_size : 64    
+ Adam optimizer with default hyperparameters    
+ $L_2$ regularization on the wieght scale parameters with coefficient $5 \cdot 10^{-5}$      

### Result   

+ Table 1. 에서 볼 수 있듯이, bits/dim 은 Pixel RNN 기준보다는 개선되지 않았지만, 다른 생성 모델과 경쟁력 있는 결과를 보여준다.    

+ 파라미터 수가 증가할수록 모델 성능이 좋아진다는 것을 알았다.    

+ CelebA dataset의 경우 완전히 이상한 smaple을 출력하기도 한다.   

+ 다만, VAE와 달리, RealNVP는 전역적으로 일관되면서도 선명하다.   

+ 저자의 가설은 실제 NVP가 낮은 주파수 구성 요소보다 높은 주파수 구성 요소를 더 강조하는 L2 노름과 같은 고정된 형식의 재구성 비용에 의존하지 않기 때문이다.     

+ Auto regressive model과 달리, RealNVP 모델에서의 샘플링은 입력 차원을 병렬로 처리하기 때문에 매우 효율적으로 수행된다.      

+ Imagenet과 LSUN에서 우리 모델은 배경/전경 개념과 조명 상호작용(밝기, 일관된 광원 방향)을 잘 캡처한 것으로 보인다.   

-------------------------------------------------------------          

## Discussion and Conclusion    

+ 논문에서 제안한 RealNVP 모델은 비선형 변환과 반전(affine coupling) 레이어를 결합하여 고차원 데이터를 모델링하는 강력한 생성 모델임을 입증했다.    

+ 생성된 샘플은 주요한 이미지 데이터셋에서 고품질로 나타났으며, 이미지의 지역적 상관 관계와 구조를 효과적으로 학습하였다.     
  - 이와 함께, RealNVP는 생성된 샘플이 전역적으로 일관되고 선명하게 나타나는 특징을 보여주었다.    

  - 특히 Imagenet과 LSUN 데이터셋에서는 배경/전경과 조명과 같은 중요한 시각적 개념을 잘 학습한 것으로 나타났다.     

+ 비선형 변환을 사용하면 복잡한 분포를 모델링할 수 있으며, 반전 레이어를 통해 조건부 밀도를 계산하기에도 효과적이다.      
