#Image Retrieval and Classification
Content Based Image Retrieval using the SIFT (Scale Invariant Feature Transform) and Bag of Features based approach to retrieve similar content images from a large database, efficiently and accurately. 

We show that a text retrieval system can be adapted to build a content  image retrieval solution. This helps in achieving scalability.


#Procedure
We represent the word images as a histogram of visual words present in the image. Visual words are quantized representations of local regions, and for this work, SIFT descriptors at interest points are used as feature vectors. The images are classified into classed and a Support Vector Machine (SVM) based trainer and predictor is used to predict class of an image. Further, other low-level features such as SURF are taken and the performance is compared with SIFT.
