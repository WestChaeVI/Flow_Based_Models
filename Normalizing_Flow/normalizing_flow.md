# [Variational Inference with Normalizing Flows](https://arxiv.org/pdf/1505.05770.pdf)       

--------------------------------------------------------------------------------------------------------       

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/b52d200b-3570-4531-b6be-59556f6b9126'height='500'></p>      

--------------------------------------------------------------------------------------------------------        

## Motivation     

+ 기존의 Variational Inference method는 주어진 입력 데이터를 저차원 manifold의 latent space에 확률 분포로의 mapping을 수행한다.     
  - 우리가 tractable한 미지함수 $q_{\phi}(\cdot)$ (e.g. Gaussian distribution)을 prior distribution으로 **가정**하고 **ELBO**(**Evidence Lower Boundary**)와 **KL Divergence**를 이용하여 posterior distribution을 추정하는 방식이다.    
  - 그러나, tractable한 미지함수 $q_{\phi}(\cdot)$ (e.g. Gaussian distribution)을 prior distribution으로 가정하는 것부터 이로인해 적절한 posterior distribution를 추정이 가능하긴 하지만 '적절한'에 focus를 맞추게 되면 쉽지 않다는 문제점을 가지고 있다.      

  - 또한, latent space에서의 연산이 어려울 수도 있고 결과적으로 생성된 sample의 quality가 제한될 수 있다.     

+ Normalizing flow는 이러한 기존의 posterior distribution를 추정하는 방식이 가진 문제점들을 **다른 접근 방식**으로 개선해 나간다.     

+ Normalizing flow는 latent vector $z$의 확률 분포에 대한 일련의 역변환(a sequence of invertible transformations)을 통해 데이터 $x$의 분포를 명시적으로 학습하며 이를 간단하게 negative log-likelihood로 해결한다.     

--------------------------------------------------------------------------------------------------------    

## Contribution    

1. A series of invertible mappings을 통해 확률 밀도를 transforming하여 complex distribution을 만들어내는 normalizing flow를 사용하여 approximate posterior distributions를 찾는 법을 제안한다.   

--------------------------------------------------------------------------------------------------------      

## Recap     

### Change of Variable     

+ 어떠한 random variable $x$와 $z$에 대해 각각 다음과 같은 **확률 밀도 함수**(**Probability Density Function,  PDF**)가 있다고 가정해보자.     

$$x \ \sim \ p(x)$$      

$$z \ \sim \ \pi(z)$$     

+ $z$가 $x$를 잘 표현하는 latent variable이고, $z$의 확률 밀도 함수가 주어진다면, 일대일 mapping 함수 $x = f(x)$를 사용해서 새로운 random variable을 구할 수 있지 않을까 기대해볼 수 있다.    

+ 그리고, 그 함수 $f$가 **invertible**하다고 가정한다면, $z = f^{-1}(x)$ 도 가능할 것이다.   

+ 그럼 이제 해야 될 것은 우리가 모르는 $x$의 확률 분포 $p(x)$를 구하는 것이다.    

### Formula Derivation    

$$\int p(x)dx \ = \ \int \pi{(z)}dz \ = \ 1$$      

$$\int p(x)dx \ = \ \int \pi{(f^{-1}(x))}df^{-1}(x)$$     

$$p(x) \ = \ \pi{(x)}\lvert\frac{dz}{dx}\rvert \ = \ \pi{(f^{-1}(x))}\lvert \frac{df^{-1}}{dx}\rvert \ = \ \pi{(f^{-1}(x))}\lvert (f^{-1})'(x)\rvert$$     

+ 위 식을 통해서, 우리가 알지 못하는 $p(x)$를 $z$의 확률 밀도 함수로 표현할 수 있게 되었다.   

+ 이식을 조금 더 직관적으로 설명하자면     
  > 서로 다른 변수 $x$, $z$의 밀도 함수들 간의 관계는 $\lvert (f^{-1})'(x)\rvert$ 만큼의 비율을 갖는다고 볼 수 있다.    

+ 사실, 우리가 실제로 다뤄야 하는 변수들은 **고차원 변수**들이기 때문에 위의 식을 **다변수**(**Multi-variable**)관점으로 다시 표현해줄 필요가 있다. $\rightarrow$ 즉, matrix로 표현하고자 한다.     

$$Z \ \sim \ \pi(Z), \ X \ = \ f(X), \ Z \ = \ f^{-1}(X)$$    

$$p(X) \ = \ \pi{(Z)}\lvert \det \frac{dZ}{dX}\rvert \ = \ \pi{(f^{-1}(X))}\lvert \det \frac{df^{-1}}{dX}\rvert$$     

+ 아직 하나 남은 것이 있다. 바로 미분이다.    
  - 행렬을 다루기 때문에 미분도 행렬의 미분으로 표현해야 된다.    
  - 행렬의 미분은 행렬이다.     
  - 이러한 도함수 행렬을 **자코비안 행렬**(**Jacobian Matrix**)이라고 한다.    


### Jacobian Matrix and Determinant    

$$ J \ = \ \frac{dY}{dX} \ = \ \left\[\begin{matrix}
\frac{\partial{y_1}}{\partial{X}} \\
\vdots \\
\frac{\partial{y_m}}{\partial{X}} \\
\end{matrix} 
\right\] \ = \left\[\begin{matrix}
\frac{\partial{y_1}}{\partial{x_1}} & \cdots & \frac{\partial{y_1}}{\partial{x_n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial{y_m}}{\partial{x_1}} & \cdots & \frac{\partial{y_m}}{\partial{x_n}} \\
\end{matrix} \right\]$$     


