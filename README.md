# Thesis-Chalkboard
Planning and Development of Super-Pixel Transformer. Will be cleaned up, or offloaded to separate repository upon 
completion. This repo will be the groundwork for multiple iterations and tests of different models. We will build our 
models with SLIC and Segment Anything, and we will test pre-trained, and trained-from-scratch transformers. 

## The Plan
Build a model that preprocesses an image with SLIC, extracts a superpixel index map, convolve each super pixel to create 
feature vectors, and then pass those vectors as tokens into a transformer for classification. 

## The Holy Whiteboard
The following whiteboard acts as the high-level blueprint for our research project. The goal is to take images that have 
been broken up into an arbitrary number of superpixels and convert those superpixels into standardized vectors for 
a transformer. 
![IMG_1128.jpg](ReadMe_Images%2FIMG_1128.jpg)
### Superpixel Algorithms
We are currently considering SLIC and Segment Anything for generating superpixels in our model. SLIC is a 
straightforward method that takes an image and the desired number of SPs(Superpixels) and generates a corresponding 
image with SP boundaries labeled. Segment Anything takes it a step further by grouping the SPs based on similarities to 
truly segment the image. This has the advantage of allowing us to train our model on more complicated tasks derived from 
segmentation, but makes it redundant to test a model on solely segmentation (cause the image is already segmented).
### Converting Superpixels into a valid input for a transformer
Unfortunately, not all superpixels are made equal. This is problematic because Transformers require uniform tokenized 
inputs to map attention between all input tokens in an image. So how do we standardize our superpixels for input? There
are several potential ways, but the one we are focused on involves convolving each superpixel in each image and taking the
feature vector from our convolution network and using said vector as the input for our transformer. A rough idea of this
concept is demonstrated below. However, a glaring issue with this process is the performance cost. A paramount advantage 
of transformers vs other network types is their computational efficiency at large scales. If we stop to both generate 
superpixels and convolve all of those superpixels for every image during training, then the time efficiency of our model 
might suffer severely. 
![IMG_1133.jpg](ReadMe_Images%2FIMG_1133.jpg)
### The End of the line
Frankly, if we can vectorize our superpixels into truly USEFUL information for our transformer, and the transformer
provides promising results with that input, then we have accomplished our largest PRACTICAL implementation goal for this research.
The next step will be demonstrating why this methodology is valuable by seeing what results we get on a spectrum of Computer
Vision tasks as well as benchmarking our model against others like it, and the industry standards. This will require looking
at the tasks and datasets our 'competitors' are using and testing our model iterations on them.
## The General Plan (From the Thesis Perspective)
Most of our version 1 model will be built using SLIC on the Oxford Pets Dataset for classification, once this iteration 
is completed, we will branch out into other iterations and tests. 
![IMG_1135.jpg](ReadMe_Images%2FIMG_1135.jpg)
## Major Goals:
- [ ] Create a functioning SP Transformer Model
  - [x] Get a functioning transformer that classifies the Oxford Pets Dataset.
  - [x] Run the transformer with SLIC in preprocessing.
  - [ ] Create a mini CNN used to convolve and pool the superpixels into standardize feature vectors.
  - [ ] Pass vectorized superpixels to Transformer for classification.
  - [ ] Ensure and optimize competitive results for our SP Transformer.
- [ ] Justify the model's Existence
  - [ ] Create a Plan based on related works for tasks and datasets to compare on.
  - [ ] Test model on different tasks, collect results with an emphasis on accuracy and efficiency.
    - - [ ] Test the model with Segment Anything.
