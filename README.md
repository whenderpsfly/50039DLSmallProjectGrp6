# 50.039 Deep Learning Y2021 Small Project
This is a project to classify X-ray images of normal, infected and covid. 

FILES:  
Multiclass.ipynb: A file to train a multiclass classifier from scratch.  
Multiclass_loadModel.ipynb: A file to load a pretrained multiclass classifier and validate it.  
multiclass_cnn.pt: Weights for pretrained multiclass classifier, for reproducibility.
Binaryclass.ipynb: A file to train a binary-class classifier from scratch.  
Binaryclass_loadModel.ipynb: A file to load the pretrained binary-class classifier and validate it.  
binaryclass_cnn.pt: Weights for pretrained binary-class classifier, for reproducibility. (2 of them)  

[toc]

## Report


![](https://i.imgur.com/Fih68ZG.png)
Graph displaying number of files in our raw dataset 

The data is split into train, test and validation. A notable observation is that the training data is unbalanced, which we will address it later on.

This is the breakdown for the number of files:
* Training:
    * Normal: 1341
    * Infected (non-covid): 2530
    * Infected (covid): 1345
* Testing:
    * Normal: 234
    * Infected (non-covid): 242
    * Infected (covid): 139
* Validation:
    * Normal: 8
    * Infected (non-covid): 8
    * Infected (covid): 8
                   
                   
#### Data Pre-Processing
![](https://i.imgur.com/KWI7W3F.png)
From the image above, we can see that the original x-ray images (top 8 images) comes in different brightness and contrast. For some images, the hearts and spines are also indistinguishable from each other. The differences in brightness and lack of contrast may result in the model being unable to classify certain images accurately. If there are less images of lower brightness like shown above, the classifier may not perform as well for those images since there are less examples to train on. Hence, we applied histogram equalization on the dataset.

After applying histogram equalization, we observe that the images (bottom 8 images) are much similar in their brightness and have more contrast. Additionally, when we compare the before and after versions of the image, "0.jpg", we see that originally the x-ray was very bright, but after histogram equalization is applied, the organs and bones appear more distinct.

Originally, using raw unprocessed images, the average accuracy we achieved was as such:
Test set: Average loss: 0.3576, Accuracy: 362/614 (59%)
Validation set: Average loss: 0.2519, Accuracy: 11/24 (46%)

However, using images that have histogram equalization applied, the highest accuracy we achieved is now with the same model:
Test set: Average loss: 0.3164, Accuracy: 395/614 (64%)
Validation set: Average loss: 0.2477, Accuracy: 13/24 (54%)

#### Mirroring data
To balance the datasets, we increased the number of images in the minority classes by mirroring the images.
As the training data is skewed to contain more infected (non-covid) images, the model would be biased towards classifying infected data input. To tackle this issue, we mirrored the 'normal' and 'infected (covid)' images. This gave us a more balanced dataset as shown below:
* Normal: 2682
* Infected (non-covid): 2530
* Infected (covid): 2690

![](https://i.imgur.com/EYRTVft.png)

After we added mirroring to the data, the highest accuracy we achieved was:
Test set: Average loss: 0.3089, Accuracy: 426/614 (70%)
Validation set: Average loss: 0.2192, Accuracy: 20/24 (83%)


#### We chose the 3-class classifier because:
1) In our case scenario, the possible classifications are mutually exclusive classes, hence it is defined as a multi-class problem.
2) Also, we feel that 2 layers may cause additional errors, i.e if the first binary classifier misclassifies as normal then the image will not go into the second binary classifier, which is a guaranteed misclassification.
3) Stacking layers of classifiers tends to slow down the training process as well as the prediction time. 
4) Most importantly, 2 binary class classifiers will have more parameters, thus we will need much more training data or else the model will overfit easily, hence negatively affecting the accuracy. 



#### Choice of model architecture:
1) Convolution layer: The x-ray image can have the same feature in many different parts of the image and all these features should be extracted concurrently. Thus, the convolution operation is performed to extract these features by sliding a kernel over the image. As the kernel slides, it computes every point of the output by weighting and summing every region of the input. A non-linear ReLU activation function is added after every convolution layer to prevent vanishing gradient during backpropagation. Example of features are directed edges and corners, gradient orientation etc. 
2) 2D max pooling layer: This layer extracts out the maximum value in every region of the image. It serves to improve image quality by suppressing noises as well as to reduce the required computation power through dimensionality reduction.
3) Dropout layer: Dropout is a regularization approach that approximates the neural networks by randomly dropping some nodes during training. It results in a noisier training process and make the model more robust. It also prevents overfitting, which is probable due to our small dataset.
4) Typical batch sizes are usually within 2-64, and the larger the batch size the faster the computation. However, we chose a batch size of 16 due to computational limits (a batch size of 64 would return CUDA not enough memory errors.) (https://arxiv.org/abs/1206.5533) We have also tested out several batch sizes but we found 16 to be optimal. 
5) Why the loss function and parameters? We chose the Cross Entropy Loss function as it is commonly used for multi-class classification problems. This is partly due to the fact that it utilises softmax to calculate the output probabilities, which we use to measure the distance from the truth values. We then try to minimise that loss to ensure that the adjustment to the weights can be done to ensure a higher accuracy next time.
6) Why optimizer and parameters? We chose the Adadelta optimizer, which is an extension of Adagrad. Instead of keeping all the past squared gradients, it restricts the window of accumulated past gradients to a fixed size *w*. Also, it stores the previous squared gradients efficiently by recursively defining a decaying average of all past squared gradients. The running average then only depends on the previous average and the current gradient. (https://ruder.io/optimizing-gradient-descent/index.html#adadelta)

7) Why initialization? For initialization, we used random initialization of the weights to ensure we break symmetry and ensure much better accuracy. However, to ensure reproducibility, we have a random seed to ensure that the model weights are reproducible. 
8) We have also recreated the model architecture in a more visual format.
![](https://i.imgur.com/esGJPH5.png)
![](https://i.imgur.com/TCWwz7E.png)


---
### Binary Classifier

#### Data pre-processing
We also created a binary classifier to compare its accuracy against our multiclass classifier.

Following the project handout, our first binary classifier was designed to classify X-ray images of normal people and people with infected lungs.
Our second binary classifier was designed to classify X-ray images of COVID/non-COVID pneumonia.

To do so, we first had to create a new folder for all images of infected patients, leaving us with 2 classes: normal and infected.

Similarly, we applied histogram equalization on the data and mirrored the normal images in the training set in order to get a more balanced dataset.

#### Distribution of images, normal vs infected
![](https://i.imgur.com/ijPh71b.png)

The initial distribution of the images:
* Training:
    * Normal: 2682
    * Infected: 3875
* Testing:
    * Normal: 234
    * Infected: 381
* Validation:
    * Normal: 8
    * Infected: 17

#### Distribution of images, covid vs non-covid
![](https://i.imgur.com/h6JyC4q.png)

The distribution of images for covid vs non-covid:
* Training:
    * Covid: 2690
    * Infected: 2530
* Testing:
    * Covid: 139
    * Infected: 242
* Validation:
    * Covid: 9
    * Infected: 8


#### Results
Model 1 (Compare normal vs infected):
- Validation accuracy: 92% (23/25)

Model 2 (Compare covid vs non-covid):
- Validation accuracy: 76% (13/17)

Hence, we concluded that the multi-class classifier performs better on average as compared to the two-layered binary classifier. 

-----------------------------------------------------------
**Question**: You might find it more difficult to differentiate between non-covid and covid x-rays, rather than between normal x-rays and infected (both covid and non-covid) people x-rays. Was that something to be expected? Discuss.

**Answer**: This result was expected. Even doctors with years of experience have difficulties in differentiating covid-19 x-ray and pneumonia x-ray, due to the high similarity in their images. Also, normal x-rays look considerably more different when compared to x-rays that are infected (both covid and non-covid), as compared to x-rays with covid and pneumonia. It would be difficult for a neural network model to pick up something that is difficult to human professionals. 
https://www.cebm.net/covid-19/differentiating-viral-from-bacterial-pneumonia/
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7306947/


-----------------------------------------------------------
**Question**: Would it be better to have a model with high overall accuracy or low true negatives/false positives rates on certain classes? Discuss.

**Answer**: In the context of covid-19, it would be better to have a model with low false negatives on covid-19 infection. False negative implies that the subject is infected with covid-19 yet, not identified by the model. This may result in disastrous effects as the infected subject will go on into the community and spread the virus to others. 

Also, false positive is also undesirable as it implies that an uninfected subject will be wrongly quarantined and they may possibly get infected due to closer contacts with other infected subjects. However, the damage is minimized to only the false positive and not spread into the community.  


-----------------------------------------------------------

**Question**: How do doctors diagnose infections based on x-rays? Does our AI seem to be able to reproduce this behavior correctly?

**Answer**: Doctors diagnose infections by looking for white patches over parts of the lungs in the X-ray image. The larger the patches, the more serious the infection. To a large extent, our AI model is able to reproduce the behaviour as the convolution networks are able to pick up different features of the patches during training. Max pooling allows us to remove the noise, or rather those area which are not white patches. Also, histogram normalization have the effect of making the images clearer and thus, the white patches will be more distinct.

However, we could further improve our AI by using semantic segmentation techniques, in which the white patches are especially highlighted and labelled, to further help the doctors. 

![](https://i.imgur.com/56KDts7.png)

The picture above shows an example of a x-ray of an uninfected person being classified as infected by the model. Although the classification was erroneous, it can be seen that there are indeed some white patches in the image. This means that our AI is able to reproduce the behavior of doctors to a large extend.