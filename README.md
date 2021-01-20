# SceneCRF

_Scene Segmentation with Conditional Random Fields_

## Intro

Scene segmentation is the task of partitioning an image into regions corresponding to the objects in them and assigning the correct class labels to a given region. In this particular formulation of the task, the number of class labels is known in advance, and a training dataset of real-life photographs is accompanied by a ground-truth dataset of the images colored with one of k colors, each color corresponding to a class label. Hence the scene segmentation task requires that the classifier learn to color these images with the right colors.

Scene segmentation is a structured prediction problem, since the classifier needs to output multiple predictions for a single example. However, it is not necessarily desirable to predict a class label for every pixel in the example image. The ground truth datasets may be noisy or incomplete, so it may be desirable to model these on a coarser scale. There is also likely not enough information on such a small scale to make class-based inferences. So we instead model the image with 20x20 patches.

Instead of working directly with the color channels or image intensity values, the approach outlined by Verbeek and Triggs projects the image patches into a lower-dimensional space. In this way, the feature detector responses on image patches function as a dimensionality reduction technique. By clustering the responses of the image patches together and then assigning them to centroids based on nearest neighbors, the predictor has a rough measure of similarity between patches of the image. The training set allows the classifier to build a visual dictionary of words, so that it scan a new image and ’look up’ the visual words in a new image to make predictions.

The importance of using a Conditional Random Field to model the object class labels becomes apparent when considering how to model spatial regularities and context in the image. A CRF models the patches as random variables with edges to each of its patch-neighbors, and an edge going to the latent class label variable which is to be predicted. This conditional independence structure can help weight predictions based on the predictions of all the other neighbors, and so help keep objects connected together despite small textural differences. For example, different parts of a connected object might not have the same visual words, and so a naive classifier would classify them as different objects, but we realize that they are the same when considering how different they are from the background or another object in the image.

## Data

I used the Microsoft Research Cambridge (MSRC) 9-class label object recognition dataset. This dataset consists of 240 images and ground truth class labels, all images having the same aspect ratio and 320x200 resolution.

The contents of the image is mostly outdoor scenes, with many pictures of animals and vehicles. It is not entirely outdoors and there are a few pictures of humans, but for the most part the pictures are of one object and not more complicated scenes. While this dataset might seem small and relatively simple, I believe it is a good match for a Conditional Random Field, because it would be far too small for a more complex classifier like a Convolutional Neural Network. Still, a fairly accurate classifier could be learned on a small dataset like this.


## Features

To build features, I decomposed the image into 20x20 patches to build a feature vector. For each patch, I compute the 128-dimensional SIFT descriptor vector using OpenCV, a 36-dimensional color hue descriptor vector, and a position vector indicating which patch it belongs in. I encoded the position vector by overlaying a 16x10 grid on the image, so each patch would be in one of 160 tiles.

As mentioned above, the classifier does not work directly with the feature responses. For each of these features, I cluster the feature response from every patch in the entire training set together, and the actual features are indicator vectors which designate the centroid assignment for that patch. This allows us to classify a patch based on other most similar patches.

I used K-means clustering to cluster the training image patches and build a visual word dictionary. The feature detectors contain 1000 words, the hue detectors contain only 36 words, and the position vectors contain 170 possible words.

## Results

I looked at the paper about SIFT descriptors, and while I ended up using some code to compute it I tried a few times to implement it myself. I was a little confused by the hue descriptors, since I have not seen many mentions of it in other papers or libraries. I ended up translating some code I found from MATLAB to compute the hue descriptor response.

I think the visual words approach is really cool and elegant, and I like the way it projects the data down to lower dimensions, encoding the relevant information for the classifier. I think that is one of the more interesting things I learned in general this quarter, that is, how images follow certain probabilistic regularities and contain information, but in computer vision we need to find ways to encode that information in a more structured way.
