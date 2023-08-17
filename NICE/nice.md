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



----------------------------------------------------------------------------------------------------       

----------------------------------------------------------------------------------------------------       

----------------------------------------------------------------------------------------------------        
