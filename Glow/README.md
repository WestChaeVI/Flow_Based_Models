# [Glow: Generative Flow with Invertible 1x1 Convolutions(2018)](https://arxiv.org/pdf/1807.03039.pdf)      

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/31f46a68-f902-4c3c-8910-f9232819231c'</p>     

---------------------------------------------------------        

## Motivation    

+ Two major unsolved problems in the field of machine learning    

  - **Data-efficiency**    
    > 인간이 제한된 정보만으로 새로운 것을 배울 수 있는 것처럼,  few datapoints 만으로 학습 수 있는 능력     

  - **Generalization**     
    > 모델이 훈련 중에 특별히 확인하지 못한 작업이나 input에 대해 잘 수행되어야 함을 의미.      
    >      
    > 현재 AI 시스템은 훈련받은 것과 다른 입력에 직면할 때 종종 어려움을 겪는다.    

+ High-dimension image 생성의 어려움     

  - 고해상도 이미지는 픽셀 수가 많아 이미지의 차원이 크기 때문에 생성 모델링이 어렵다. 고해상도 이미지에서는 **픽셀 간의 공간적인 의존성과 다양한 구조가 중요하게 작용**하기 때문에 모델이 이를 잘 처리할 수 있어야 한다.    

+ 확장성과 효율성의 필요성     

  - 이전에 개발된 몇몇 생성 모델은 고해상도 이미지 생성에서의 확장성과 효율성을 제공하기 어려웠다. 이 모델들은 큰 이미지에 대한 생성 작업에서 **메모리 및 계산 요구 사항이 매우 높은** 문제가 있었다.   

---------------------------------------------------------        

## Introduction    

+ 생성 모델의 연구, 특히 likelihood-based method를 사용하는 모델들, 그리고 GANs은 최근(2018) 몇 년간 엄청난 발전이 있어왔다.    

+ Likelihood-based methods는 세 가지 categories로 나눠볼 수 있다.    

  - **Auto-Regressive models**     
    > 자기회귀모델은 simplicity를 이점으로 가진다.     
    >       
    > 하지만, synthesis(합성)에는 parallelizability(병렬성)이 제한되어 있다는 단점이 있다.     
    >       
    > 계산 비용이 데이터의 차원에 비례하기 때문에, 큰 이미지나 비디오 데이터는 다루기 어려운 문제가 있다.    

  - **Variational autoencoder(VAEs)**     
    > ELBO(Evidence of Lower Bound) term을 사용하여 optimize한다.    
    >      
    > VAE는 training과 synthesis의 parallelizability가 가능하다는 이점이 있다.       
    >       
    > 그러나, optimize하는 작업이 비교적 어렵다는 것이 문제이다.     

  - **Flow-based generative models**     
    
    - **Exact latent-variable inference and log-likelihood evaluation**     
      > VAEs는 오직 datapoint에 해당하는 latent space를 approximation을 통해 추론한다.      
      > GANs는 latents를 추론하기 위한 encoder가 전혀 존재하지 않는다.     
      >       
      > **In reversible generative models**, 이 모델들은 approximation 없이도 추론이 가능하다.     
      > 정확한 추론을 하게 할 뿐만 아니라, data의 exact log-likelihood를 최적화를 수행할 수 없다. (lower bound를 사용하지 않고)    

    - **Efficient inference and efficient synthesis**     
      > PixelCNN과 같은 autoregressive model들은 또한 reversible하다        
      >     
      > 그러나 합성은 병렬화하기 어렵다. 그리고, 전형적으로 비효율적이다.     
      >       
      > Glow(and RealNVP)와 같은 flow-based generative models은 추론과 합성 들다 병렬화가 효율적이다.    

    - **Useful latent space for downstream tasks**      
      > autoregressive model들의 hidden layer들은 unknown marginal distribution을 가진다.    
      >     
      > 근데 이러한 marginal distribution은 data의 manipulation을 수행하기 어렵게 만든다.      
      >      
      > In GANs, datapoints는 종종 latent space로 바로 표현되지 않을 수 있다. 그 이유는 encoder가 없고 데이터 분포를 완벽하게 지원하지 못할 수 있기 때문이다.    
      >     
      > VAEs의 경우는 reversible generative model이 아니다. VAEs는 datapoint간의 interpolation 및 기존 datapoint의 의미 있는 modification과 같은 다양한 응용 프로그램을 허용한다.      

    - **Significant potential for memory savings**     
      > reversible한 neural network에서의 gradient 계산은 깊이가 선형이 아닌 일정한 메모리를 요구한다. (자세한 설명 [RevNet paper](https://arxiv.org/abs/1707.04585)    

---------------------------------------------------------        

## Proposed Generative Flow    

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/61f24401-4be6-4343-b916-f4ddc18c77bc'</p>     

### Actnorm: scale and bias layer with data dependent initialization    

+ actnorm은 Batchnorm 처럼 잘 학습시키기위한 normalization을 해주는 모듈이다. batchnorm이 가지고 있는 문제점을 해결하기 위해서 제안한 normalization 기법이다.    
  > Batch Normalization의 문제점      
  >           
  > mini-batch size가 작으면, activation의 noise의 분산이 커지는데     
  > 이는 gradient vanishing/explosion, 학습속도를 야기할 수 있다.    
  >       
  > 그러나, 모델을 학습시킬 때 메모리는 한정되어 있기 때문에 dataset이 큰 경우 보통 mini-batch를 1로 잡고 돌린다.     
  > 즉, 대용량 데이터는 batch normalization을 수행하면 성능이 저하된다.    

+ 다시 말해, activation의 노이즈의 분산이 크기 때문에 batchnorm의 성능이 저하되는데, 큰 이미지를 프로세싱하기 위해서는 GPU의 한계 때문에 minibatch의 크기를 줄여야 한다는 문제가 있었다.    

+ actnorm은 이를 해결하기 위해 **초기 미니배치 데이터에 대해 채널별로 평균이 0, 분산이 1 을 갖도록 초기화**를 시킨다.    

$$\forall i, j : y_{i, j} \ = \ s \odot x_{i, j} \ + \ b$$     

+ 각 채널별로 scale $s$와 bias $b$, 즉 평균이 0, 분산이 1 을 갖도록 초기화한 trainable parameter를 이용하여 affine transformation을 수행한다.      



### Invertible 1 × 1 convolution      

+ NICE, RealNVP등의 flow based generative model에서 feature를 섞어주기 위해 channel의 순서를 반대로 했던 방법을 고도화한 것이다.   

+ Glow에서는 *1x1 conv* 연산을 통해 **permutation**의 효과를 냈다.   

+ flow를 적용하기 위해 이 convolution의 jacobian matrix의 log determinant를 구해야하는데, 결론적으로 weight(C x C)를 갖는 matrix $W$의 log determinant에 곱하려는 tensor의 $h$, $w$를 dot product한 형태이다.     

The log-determinant of an invertible 1 × 1 convolution of a h × w × c tensor **h** with c × c weight
matrix **W** is straightforward to compute:     

$$\log \lvert \det ( \frac{\partial \ \text{conv2D}(h; W)}{\partial \ h} ) \rvert \ = \ h \cdot w \cdot \log \lvert 
\det (W) \rvert$$   

+ 그러나 위 식의 $\det (W)$ 시간 복잡도는 $\mathcal{O}(c^3)$이다. 효과적인 연산을 수행하기 위해서는 이 복잡도를 줄여야한다.    

+ triangular matrix의 determinant는 대각성분의 합이라는 점에서는 착안하여 **LU decomposition**을 이용하여 constant한 시간 복잡도로 이를 해결하였다.    


#### LU Decomposition    

+ $\det (W)$의 시간 복잡도 $\mathcal{O} (c^3)$를 $\mathcal{O} (c)$로 바꾸기 위해 사용한다.   

$$\text{W} \ = \ \text{PL}(\text{U} \ + \ diag(\text{s}))$$    

+ P : permunation matrix    
+ L : Lower triangular matrix with ones on the diagonal    
+ U : Upper triangular matrix with zeros on the diagonal   
+ s : vector   

그러면 log-determinant는 단순하게 나온다.    

$$\log \lvert \det (\text{W}) \rvert \ = \ \text{sum} ( \log \lvert \text{s} \rvert)$$    

+ 저자는 실험을 통해 시간의 큰 차이는 없었지만 계산 비용은 $c$에 따라 중요하게 작용한다고 언급했다.    

### Affine Coupling layers    

+ 이전의 NICE, RealNVP에서 invertible, 간단한 Jacobian determinant 계산의 조건을 만족하는 affine coupling layer를 소개했다.    

+ Glow에서도 이와 동일한 방법을 사용한다.    

+ affine coupling layer에는 **세 가지 기능**으로 구성되어 있다.

#### zero initialization   

+ affine coupling layer가 동일한 함수를 여러번 수행하기 위해 각각 NN의 마지막 convolution의 weight를 0으로 초기화한다.     
  > 매우 깊은 network를 학습시킬 때 도움이 된다고 한다.   

#### split and conatenation     

+ Tensor를 반으로 나누고 합치는 기능을 말한다.     

+ split, concatenation을 함수로 본다면, 둘은 역함수 관계이다.     

#### permutation    

+ flow의 step 단계는 각 차원이 다른 모든 차원에 영향을 미칠 수 있도록 변수에 대한 permutation이 선행되어야 한다.    

+ NICE, RealNVP는 단순히 차원의 순서를 반대로 하는 것으로 permuation을 수행했었다.    

+ Glow에서는 invertible 1x1 conv로 이 기능을 수행하였다.     


<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/3038fbca-d104-4980-a449-33decdf5166d'</p>     

---------------------------------------------------------        

## Quantitative Experiments    

+ RealNVP 모델과 비교하기 위해 여러 데이터셋을 가지고 실험을 진행하였다.     

### Optimization details    

+ Adam ($\alpha = 0.001$, $\beta_1$, $\beta_2$ is default)    

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/12f322df-8581-4749-8b5b-3f787daab686'</p>     

#### Gains using invertible 1 x 1 Convolution    

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/ec610261-371d-4b33-a547-999e4ad90ef6'</p>    

+ CIFAR-10 dataset에 대해서 additive coupuling과 Affince coupling에 대해서 NLL을 비교하였고    

+ Channel variables의 permutation에 대해 **세 가지 variations**을 실험했다.    
  > RealNVP의 reversing operation (파란색)      
  > Fixed random permutation (초록색)    
  > ours (invertible 1 × 1 conv, 빨간색)    

#####  결과 해석   

+ additive, affine coupling 모두 Ours(invertible 1x1 conv)가 더 낮은 NLL를 가지며 더 빠르게 수렴하는 것을 볼 수 있다.   

+ 또한, additive와 affine coupling을 비교하였을 때는 affine coupling이 더 빠르게 수렴한다.    

+ 저자는 invertible 1x1 conv model의 wallclock time 증가가 단지 약 7%이며, 효율적이다라고 언급했다.     
  > wallclock time : 실제 코드를 실행하는 데에 걸린 시간      

#### Comparison with RealNVP on standard benchmarks      

+ 저자가 제안한 아키텍처가 전반적으로 RealNVP 아키텍처와 경쟁력 있는지 확인하기 위해, 다양한 데이터셋에서 모델을 비교했다.    
  > Dataset : CIFAR=10, ImageNet, LSUN     

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/742bb82b-d801-43c9-8b08-a0fc3ec3e274)'</p>     

#####  결과 해석     

+ 모든 dataset에 대해서 RealNVP보다 더 좋은 성능이 나온 것을 볼 수 있다.    

---------------------------------------------------------        

## Qualitative Experiments    

+ 모델이 고해상도에 대응할 수 있는지, 실제적인 샘플을 생성할 수 있는지, 의미 있는 잠재 공간을 생성할 수 있는지를 연구하고자 실험을 진행.     

### Optimization details    

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/c8b41ba2-7baa-4447-a83d-d3833f46629a'</p>    

+ 저자는 실험을 통해 reduced-Temperature model에서 sampling하는 것이 종종 고품질 샘플을 얻는 데 도움이 된다는 것을 발견하였습니다.      
  > temperature $T$로 sampling할 때, 분포 $p_{θ,T(x)} ∝ (p_θ (x))^{T^2}$에서 sampling한다.    
  >       
  > additive coupling layer의 경우에는 $p_θ (z)$의 표준 편차에 $T$의 배수를 곱함으로써 이를 달성할 수 있다.     >      
  > **Temperature** : 생성 모델에서 샘플링을 조절하는 hyper-parameter이다. 온도는 확률 분포의 확률 값을 변화시켜 샘플의 다양성과 품질을 조절하는 데 사용된다.    
  >     
  > **더 높은 온도 (High Temperature)**: 높은 온도로 샘플링하면 모델이 더 다양한 샘플을 생성하게 된다. 확률 분포의 뾰족한 부분을 평평하게 만들어서 다양한 결과를 얻을 수 있게 해준다. 이는 생성된 샘플들이 다양한 형태나 변화를 보이는 데 도움이 된다.     
  >    
  > **낮은 온도 (Low Temperature)**: 낮은 온도로 샘플링하면 모델이 더 확실한 예측을 하게 된다. 확률 분포의 뾰족한 부분이 더 뾰족하게 되어, 가장 높은 확률 값을 가진 샘플이 선택될 가능성이 높아진다. 이는 생성된 샘플들이 더 일관된 특징이나 형태를 가지게 해준다.     


### Synthesis and Interpolation    

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/8bf36067-af66-4dd1-8f98-3da89a320e2d'</p>    

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/f7601ff2-2289-4efe-bdcd-2ce52c4d85c8'</p>     



+ Figure 4는 Glow 모델에서 얻은 random sample을 보여준다.    
  - non-autoregressive likelihood based model의 경우 이미지의 품질이 매우 높다.    
  - 얼마나 잘 보간할 수 있는지 확인하기 위해 한 쌍의 실제 이미지를 가져와서 인코더로 인코딩하고 latents 사이를 선형 보간하여 샘플을 얻었다.      
  - Figure 5의 결과는 generator distribution의 image manifold가 매우 smooth하고 거의 모든 중간 sample이 실제 얼굴처럼 보이는 것을 볼 수 있다.

### Semantic Manipulation     

+ 이미지의 attributrs를 modifying하는 것을 고려한 실험이다.    

+ 해당 attribute가 있는 이미지에 대한 평균 latent vector $z_pos$를 계산하고 attribute가 없는 이미지에 대해 $z_neg$를 계산한 후 그 차이($z_pos − z_neg$)를 interpolation 방향으로 사용했다.    

+ 이는 상대적으로 적은 양의 supervision이며 모델이 훈련된 후에 수행되므로(훈련 중에는 레이블이 사용되지 않음) 다양한 대상 attributes에 대해 매우 쉽게 수행할 수 있다고 말한다.

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/eff922bd-c4ee-462d-b01b-c6925d04f04d'</p>     

### Effect of temperature and model depth     

<table align='center'>
  <td>
    <p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/6fbbd7a7-c3dc-4063-8c2e-8821c7a9c3fb'</p>
  </td>
  <td>
    <p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/1b762d10-f59e-41a8-8169-5e67a90bcb47'</p>
  </td>
</table>    


+ Figure 8은 sample의 quality와 다양성이 어떻게 변하는지 보여준다.    
  - Temperature에 따라 달라진다.   
    > 가장 높은 온도에는 데이터 분포의 entropy를 과대평가했디 때문에 이미지에 noise가 있을 수 있으므로 sample의 다양성과 품질을 위한 최적의 지점으로 **0.7**의 온도를 선택했다.     

+ Figure 9는 모델의 깊이가 모델의 long-range dependencies를 학습하는 능력에 어떻게 영향을 미치는지 보여준다.   

---------------------------------------------------------        

## Conclusion     

+ 저자는 **Glow**라는 새로운 유형의 Flow를 제안하고 standard image modeling benchmarks의 log-likelihood 측면에서 향상된 정량적 성능을 보여줬다.    

+ 또한, 고해상도 얼굴을 train할 때 모델이 사실적인 이미지를 synthesis할 수 있음을 보여줬다.    

+ Glow 모델은 고해상도 nature image를 효율적으로 합성할 수 있는 문헌 최초의 likelihood-based model이다.    
