# Flow based models
## Flow based models paper summarization and implementation      

------------------------------------------------------------------------------------------------------------        
## [NICE(2014)](https://github.com/WestChaeVI/Flow_Based_Models/blob/main/NICE/nice.md)     

+ Dataset : [MNIST](https://paperswithcode.com/dataset/mnist)     
+ data_dim : 28 * 28    
+ hidden_dim : 1000     
+ Batch_size : 32    
+ epochs : 500      
+ optimizer : Adam     
+ lr : 0.0002      
+ weight_decay : 0.9        

### Result of Experiment     

<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/ffbcb0c4-0639-4d93-b2f7-730a5593efb4' height='600' width='600'></p>   

------------------------------------------------------------------------------------------------------------    

## [Normalizing Flow(2015)](https://github.com/WestChaeVI/Flow_Based_Models/blob/main/Normalizing_Flow/normalizing_flow.md)     

+ flow_length = [2, 8, 32]     
+ Batch_size : 4096    
+ data_dim : 2
+ iterations : 20000      
+ optimizer : RMSProp
+ lr : 1e-5      
+ momentum : 0.9      


### Table : Test energy functions  = exact_log_density  

<table align='center'width="800" height="500">
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

### Result of Experimnets     


<p align='center'><img src='https://github.com/WestChaeVI/Flow_Based_Models/assets/104747868/67b0d080-e08c-43e6-ba98-b0faa3d3ed25' height='600' width='800'></p>     

------------------------------------------------------------------------------------------------------------       

## [RealNVP(2016)](https://github.com/WestChaeVI/Flow_Based_Models/blob/main/RealNVP/realnvp.md)    


------------------------------------------------------------------------------------------------------------       

## [Glow(2018)](https://github.com/WestChaeVI/Flow_Based_Models/blob/main/Glow/glow.md)    


------------------------------------------------------------------------------------------------------------       

## [Glow-TTS(2020)](https://github.com/WestChaeVI/Flow_Based_Models/blob/main/Glow_TTS/glow_tts.md)    

