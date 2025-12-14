# Scene Reconstruction in Autonomous Vehicles

**Mini Project Report**


## Overview

This project explores scene reconstruction techniques for autonomous vehicles using Variational Autoencoders (VAEs) for image generation and interpolation. The work focuses on understanding latent space representations and their application in generating intermediate images between two input images.

## Introduction

Image generation through latent space manipulation is a crucial area of research for autonomous vehicle applications. This project investigates how Variational Autoencoders can be used to:

- Transform complex raw image data into simplified latent space representations
- Perform dimensionality reduction while preserving essential features
- Generate interpolated images between two input images
- Understand representation learning in the context of scene reconstruction

## Key Concepts

### Latent Space Representation
- **Purpose**: Compress high-dimensional image data into lower-dimensional representations
- **Process**: Lossy compression that retains relevant information while discarding noise
- **Benefit**: Enables focus on the most important features for analysis

### Variational Autoencoders (VAEs)
- Probabilistic extension of traditional autoencoders
- Regularized encoding distribution for better latent space properties
- Components:
  - Probabilistic encoder: qϕ(z|x)
  - Probabilistic decoder: pθ(x|z)
  - Regularized latent space representation

## Methodology

### Phase 1: Initial Experiments with MNIST
- Implemented basic VAE concepts using MNIST dataset
- Developed image interpolation between two handwritten digits
- Process:
  1. Encode two input numbers through the encoder
  2. Generate interpolation between encodings
  3. Decode interpolated representations to generate intermediate images

### Phase 2: Method 1 - Basic Image Transition
- Extended approach to custom dataset images
- Fitted model with same input-target image pairs
- **Challenge**: Interpolated images showed dominance of one parent image
- **Issue**: Insufficient training data led to poor results

### Phase 3: Method 2 - Comprehensive Training
- Trained model using entire dataset
- Scaled image dimensions from 28×28 to 256×256
- **Improvements**:
  - Better loss convergence with learning rate optimization
  - Enhanced image clarity through epoch tuning
  - Resolved kernel crashes for larger dimensions

## Technical Implementation

### Image Preprocessing
- Normalization of pixel values to range [0,1]
- Dataset split into training and testing arrays
- Image dimension scaling capabilities

### Architecture Components

**Encoder**:
- Input shape: (1024, 1024, 1)
- Output: 2-dimensional latent space representation
- Enables scatter plot visualization of image distribution

**Decoder**:
- Input shape: (2,)
- Output: Reconstructed image (1024, 1024, 1)
- Uses Conv2DTranspose layers for inverse transformation

**VAE Model**:
- Connected encoder-decoder architecture
- Trainable using standard fit() method
- Three interconnected models for complete functionality

## Results

The project successfully demonstrates:
- Effective latent space interpolation between images
- Quality improvements through proper training methodology
- Scalable architecture for various image dimensions
- Practical application of VAEs for scene reconstruction tasks

## Applications in Autonomous Vehicles

This work provides foundational techniques for:
- Scene understanding and reconstruction
- Intermediate frame generation for smooth transitions
- Data augmentation for training autonomous vehicle systems
- Compression of visual data for efficient processing

## Technologies Used

- **Framework**: Keras, TensorFlow.js
- **Language**: Python
- **Architecture**: Variational Autoencoders
- **Dataset**: Custom image dataset + MNIST (for initial experiments)

## Files Structure

- `Report_miniproject.pdf` - Complete project report
- `MNIST interpolation` - Initial MNIST implementation
- `Image Transition without Training.ipynb` - Method 1 implementation
- `Image Transition with Training.ipynb` - Method 2 implementation

## References

1. [Understanding Latent Space in Machine Learning](https://towardsdatascience.com/understanding-latent-space-in-machine-learning-de5a7c687d8d)
2. [Understanding Variational Autoencoders (VAEs)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
3. [Latent Space Interpolation using Keras and TensorFlow.js](https://medium.com/@noufalsamsudin/latent-space-interpolation-of-images-using-keras-and-tensorflow-js-7e35bec01c5a)
4. [Homomorphic Latent Space Interpolation for Unpaired Image-to-Image Translation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Homomorphic_Latent_Space_Interpolation_for_Unpaired_Image-To-Image_Translation_CVPR_2019_paper.pdf)

---

*This project demonstrates the practical application of deep learning techniques in autonomous vehicle scene reconstruction, providing insights into latent space manipulation and image generation methodologies.*
