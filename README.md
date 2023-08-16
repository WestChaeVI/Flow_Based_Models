# Flow_based models
## Flow_based models paper summarization and implementation

------------------------------------------------------------------------------------------------------------    

## [Normalizing Flow(2015)](https://github.com/WestChaeVI/Flow/blob/main/DCGAN/dcgan.md)     

+ Dataset : [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
+ Batch_size : 128    
+ nz : 100 (Size of z latent vector)
+ epochs : 20
+ lr : 0.0002
+ beta1 : 0.5    
<p align="center">
<img src="https://github.com/WestChaeVI/CNN-models/assets/104747868/61d00cea-c8b2-4155-8d03-114b017cc031" width="850" height="400">  
</p>     

------------------------------------------------------------------------------------------------------------       

## [SRGAN(2016)](https://github.com/WestChaeVI/GAN/blob/main/SRGAN/srgan.md)    

+ Dataset : [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
+ crop_size = 96
+ upscale_factor : 4
+ Batch_size : 64 (validset : 1)    
+ epochs : 250

------------------------------------------------------------------------------------------------------------       

## [Pix2pix(2016)](https://github.com/WestChaeVI/GAN/blob/main/PIX2PIX/pix2pix.md)    

+ Dataset : [facades](https://www.kaggle.com/datasets/balraj98/facades-dataset)
+ Batch_size : 32
+ epochs : 100 
+ lambda_pixel : 100 (Loss_func_pix weights)
+ patch : (1, 256//2**4, 256//2**4)
+ lr : 2e-4
+ beta1 = 0.5
+ beta2 = 0.999      
<p align="center">
<img src='https://github.com/WestChaeVI/CNN-models/assets/104747868/59fb009b-8140-419f-8a86-085aff830f6f' width='600' height="600">
</p>    

------------------------------------------------------------------------------------------------------------       

## [WGAN(2017)](https://github.com/WestChaeVI/GAN/blob/main/WGAN/wgan.md)    

+ Dataset : [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
+ Batch_size : 64    
+ nz : 100 (Size of z latent vector)
+ **n_critics : 5**
+ epochs : 20
+ lr : 0.0001
+ beta1 : 0.5
+ beta2 : 0.9
+ **weight_cliping_limit : 0.01**
+ **lambda_gp : 10(gradient penalty**    
