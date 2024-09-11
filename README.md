# Thesis-Chalkboard
Planning and Development of Super-Pixel Transformer. Will be cleaned up, or offloaded to separate repository upon completion. This work will be the groundwork for multiple iterations and tests of different models. We will build the models SLIC and Segment Anything and we will test pre-trained, and trained-from-scratch transformers. 

## The Plan
Build a model that preprocesses an image with SLIC, extracts a superpixel index map, convolve each super pixel to create feature vectors, and then pass those vectors as tokens into a transformer for classification. 