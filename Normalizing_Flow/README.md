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

+ A series of **invertible mappings**을 통해 확률 밀도를 transforming하여 **complex distribution**을 만들어내는 '**normalizing flow**'를 사용하여 approximate posterior distributions를 찾는 법을 제안한다.    
  > Inference with normalizing flows는 linear time complexity의 term만 추가되는 항이라 더 엄격하고, modified variational lower bound를 제공한다.     

+ infinitesimal flows를 사용하여 자주 인용되는 variational inference의 한계점을 극복한다는 것을 보여준다.    
  > infinitesimal flows : normalizing flows가 asymptotic regime에서 **ture posterior distribution $p(z)$**를 recover할 수 있는 posterior approximations의 class를 지정할 수 있게 도와준다.      

+ 특수한 유형의 normalizing flows를 적용하여 개선된 posterior approximation을 위한 관련 접근 방식의 통일된 관점을 제시한다.      

+ Posterior approxiamtion에 대하여 일반적인 noramlizing flows가 다른 경쟁 선상에 있는 접근 방식보다 체계적으로 우수하다는 것을 실험적으로 보여준다.     



--------------------------------------------------------------------------------------------------------      

## Recap     

### Change of Variable     

+ 어떠한 random variable $x$와 $z$에 대해 각각 다음과 같은 **확률 밀도 함수**(**Probability Density Function,  PDF**)가 있다고 가정해보자.     

$$x \ \sim \ p(x)$$      

$$z \ \sim \ \pi(z)$$     

+ $z$가 $x$를 잘 표현하는 latent variable이고, $z$의 확률 밀도 함수가 주어진다면, 일대일 mapping 함수 $x = f(z)$를 사용해서 새로운 random variable을 구할 수 있지 않을까 기대해볼 수 있다.    

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

--------------------------------------------------------------------------------------------------------       

## Normalizing Flow    

+ 정말 좋은 $p(X)$를 추정한다는 것(density estimation)은 쉽지 않다.     
  > 실제 딥러닝 생성 모델들은 사후 확률 $p(z|x)$를 비교적 **간단한 확률 분포로 가정**하거나 근사시킨다.     
  > 일반적으로, Gaussian distribution을 사용      
  >         
  > 실제 데이터 분포는 (i.e. $p(x)$)는 굉장히 복잡하기 때문에 적어도 latent variable의 확률 분포가 단순해야 backpropagation 계산을 조금이라도 더 쉽게 할 수 있다.    

+ 앞서 어떠한 확률 분포에 **역변환 함수**를 적용하여 새로운 확률 분포로 변환할 수 있는 것을 확인했다.   

+ Normalizing Flow는 단순한 확률 분포에서부터 일련의 역변환 함수를 적용하여 점차 복잡한 확률 분포로 변환해 나간다.     

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/4be9c0e3-24a4-4c61-846a-4017b42eae27'></p>         

+ 위 그림의 정의에 따르면 다음과 같이 나타낼 수 있다.    
$$Z_{i-1} \ \sim \ p_{i-1}(Z_{i-1})$$        

$$Z_i \ = \ f_{i}(Z_{i-1}), \ thus \ Z_{i-1} \ = \ f_{i}^{-1}(Z_{i})$$      

$$p_i(Z_i) \ = \ p_{i-1}(f_{i}^{-1}(Z_i)) \lvert \det \frac{d f_i^{-1}}{d Z_i} \rvert$$    

$$ \ \ \ \ \ = \ p_{i-1}(Z_{i-1}) \lvert \det \left( \frac{d f_i}{d Z_{i-1}} \right)^{-1} \rvert$$     

$$ \ \ \ \ \ = \ p_{i-1}(Z_{i-1}) \lvert \det \frac{d f_i}{d Z_{i-1}} \rvert^{-1}$$      

양변에 $\log$ 씌우기    

$$ \log p_i(Z_i) \ = \ \log p_{i-1}(Z_{i-1}) \ - \ \lvert \det \frac{d f_i}{d Z_{i-1}} \rvert$$      

+ Inverse function theorem and Jacobian of invertible function 특성에 따라 전개하였다.     
  
  - **Inverse function theorem**     
    > In the Inverse function theorem, 만약 $y=f(x)$ 와 $x=f^{-1}(y)$ 가 있다면,     
    > $$\frac{df^{-1}(y)}{dy} \ = \ \frac{dx}{dy} \ = \ \left(\frac{dy}{dx}\right)^{-1} \ = \ \left(\frac{df(x)}{dx}\right)^{-1}$$       
    > 가 된다. 즉, 역함수의 미분과 함수의 미분은 inverse 관계라는 것이다. 따라서 역함수의 자코비안을 함수의 자코비안의 역으로 표현이 가능하다.    

  - **Jacobian of invertible function**     
    > Jacobian of invertible function은 가역행렬인 경우, 행렬식 특성들을 갖는다.    
    > $$\det(M^{-1}) \ = \ (\det (M))^{-1}$$    
    > $$\det(M)\det(M^{-1}) \ = \ \det(M \cdot M^{-1}) \ = \ \det(I) \ = \ 1$$     

### Final Formula     

+ 앞서 다뤘던 내용들을 토대로 $Z_0$ 의 확률분포에서부터 시작해 $K$ 번의 역변환을 통해 $X$의 확률분포를 구할 수 있다.     

$$X \ = \ Z_K \ = \ f_K \ \circ \ f_{K-1} \ \circ \ \cdots \ \circ \ f_1(Z_0)$$    

$$\log p(X) \ = \ \log \pi_{K}(Z_K) \ = \ \log \pi_{K-1}(Z_{K-1}) - \log \lvert \det \frac{d f_K}{d Z_{K-1}} \rvert$$      

$$\ = \ \log \pi_{K-2}(Z_{K-2}) \ - \ \log \lvert \det \frac{d f_{K-1}}{d Z_{K-2}} \rvert \ - \ \log \lvert \det \frac{d f_K}{d Z_{K-1}} \rvert$$    

$$ \ = \ \cdots $$    

$$ \ = \ \log \pi_0(Z_0) \ - \ \sum_{i=1}^{K} \log \lvert \det \frac{d f_i}{d Z_{i-1}} \rvert$$

+ 지금까지의 방정식들이 계산 가능하게 하려면 두 가지 조건을 충족해야한다.   
  - 함수 $f$는 invertible    
  - $f$ 에 대한 jacobian determiant는 계산하기 쉬워야 한다.    

--------------------------------------------------------------------------------------------------------        
## Representative Power of Noramlizing Flows      

+ Table : Test energy functions      

<table align='center'>
  <th colspan='2'>
    <p align='center'>$U(Z)$ function</p>
  </th>
  <tr>
    <td>
      <p align='center'>$U_{1}(Z)$</p>
    </td>
    <td>
      <p align='center'>$$\frac{1}{2} \left( \frac{\lVert Z\rVert - 2}{0.4} \right)^{2} \ - \ \ln \left( e^{-\frac{1}{2} \left[ \frac{Z_{1} - 2}{0.6} \right]^{2}} \ + \ e^{-\frac{1}{2} \left[ \frac{Z_{1} + 2}{0.6} \right]^{2}} \right)$$</p>
    </td>
  </tr>
  
  <tr>
    <td>
      <p align='center'>$U_{2}(Z)$</p>
    </td>
    <td>
      <p align='center'>$$\frac{1}{2} \left[ \frac{Z_{2} - w_{1}(Z)}{0.4} \right]^2$$</p>
    </td>
  </tr>

  <tr>
    <td>
      <p align='center'>$U_{3}(Z)$</p>
    </td>
    <td>
      <p align='center'>$$- \ \ln \left( e^{-\frac{1}{2} \left[ \frac{Z_{2} - w_{1}(Z)}{0.35} \right]^{2}} \ + \ e^{-\frac{1}{2} \left[ \frac{Z_{2} + w_{1}(Z) + w_{2}(Z)}{0.35} \right]^{2}} \right)$$</p>
    </td>
  </tr>

  <tr>
    <td>
      <p align='center'>$U_{4}(Z)$</p>
    </td>
    <td>
      <p align='center'>$$- \ \ln \left( e^{-\frac{1}{2} \left[ \frac{Z_{2} - w_{1}(Z)}{0.4} \right]^{2}} \ + \ e^{-\frac{1}{2} \left[ \frac{Z_{2} + w_{1}(Z) + w_{2}(Z)}{0.35} \right]^{2}} \right)$$</p>
    </td>
  </tr>
  <tr>
    <td colspan='2'>
      <p align='center'>with $w_{1}(Z) \ = \ \sin \left( \frac{2\pi{Z_1}}{4} \right) \ $, $w_{2}(Z) \ = \ 3e^{-\frac{1}{2} \left[ \frac{Z_{1} - 1}{0.6} \right]^{2}} \ $, $w_{3}(Z) \ = \ 3\sigma \left( \frac{Z_{1} - 1}{0.3} \right) \ $, $\sigma(x) = \frac{1}{1 \ + \ e^{-x}}$</p>
    </td>
  </tr>
</table>       

+ *Figure*      
  - 완쪽 : (a) True posterior; (b) Approx posterior using the normalizing flow; (c) Approx posterior using
NICE     
  - 오른쪽 : Summary results comparing KL-divergences between the true and approximated densities for the first 3 cases.      

<table>
  <tr>
    <td>
      <p align='center'>Approximating four non-Gaussian 2D distributions</p>
    </td>
    <td>
      <p align='center'>Comparison of KL-divergences</p>
    </td>
  </tr>
  <tr>
    <td>
      <p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/f65951b2-4653-4e9c-aec0-c5a9a608bdd2'></p>
    </td>
    <td>
      <p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/55ac9d52-5cab-48f1-9e85-0bce11edb96d'></p>
    </td>
  </tr>
</table>        

+ 일반적으로 사용되는 posterior approximation으로 잡아낼 수 없는 분포(multi-modal 및 주기성과 같은 특성을 가진)를 보여주는 4 가지 examples에 대한 true distribution을 보여준다.   

+ 왼쪽 그림을 보면 알 수 있듯이 flow length를 늘리면 apprixmation quality가 크게 향상된다.    

+ NICE와 Planar Flow가 length가 증가함에 따라 동일하게 점근적인 성능을 달성하지만, Planar Flow가 훨씬 더 적은 parameter를 가지는 것을 알 수 있다.    
  > 아마도, 학습되지 않고 randomly intialized 구성요소를 혼합하기 위해 추가적인 메커니즘이 필요한 NICE와 달리     
  >      
  > flow의 모든 parameter가 학습되기 때문일 것이다.     

--------------------------------------------------------------------------------------------------------        

# Conclusion and Discussion     

+ 저자는 Normalizing Flows를 통해 simple densities에서 더 복작한 densities로의 변환을 학습하여 highly non-Gaussian posterior densities를 학숩하기 위한 간단한 approach를 개발 했다.   

+ **Inference network**와 효율적인 **Monte Carlo gradient estimation**을 사용하는 **variational inferece**를 위한 **amortized approach**와 결합하면 다양한 문제에 대한 단순한 근사치보다 더 명확한 개선을 보여줄 수 있다.     

+ Normalizing flows 관점을 사용하여 flexible한 posterior estimation을 위해 밀접하게 관련된 다른 방법들을 통합한 관접을 제공할 수 있다.    
  > flexible한 posterior estimation란?     
  > 서로 다른 통계적 or 계산적 trade-off를 통해 보다 강력한 posterior approximation을 설계하기 위한 광범위한 접근 방식     

+ Variational inference를 위한 extremely rich posterior approximations를 생성할 수 있는 normalizing flows class가 존재한다는 것이다.   
  > flow를 noramlizing하면 asymptotic regime에서 solution의 space가 true posterior distribution을 포함할 만큼 충분히 rich하다는 것을 보여준다.   

+ 이것을 latent variable model의 특정 class에서 maximum likeliehood parameter estimation을 위한 local convergence와 consistency result를 결합하면 기본 접근 방식으로 variational inferece를 사용하는 것에 대한 문제점들을 극복할 수 있다.    

