# Detection of Real vs Generated Food Images Using GANs

## Project Overview

This project implements Generative Adversarial Networks (GANs) to generate and classify food images as **real** or **fake**. Two GAN architectures were developed and compared:

- DCGAN (Deep Convolutional GAN)
- WGAN (Wasserstein GAN)

The objective is to analyze training stability, image quality, and discriminator performance, and finally build a system that accepts a user input image and predicts whether it is real or generated.

---

## Problem Statement

With the advancement of deep learning and image generation models, distinguishing between real and artificially generated images has become increasingly difficult. This project addresses the challenge of detecting fake food images using adversarial neural networks.

The system aims to:

- Generate realistic food images
- Train a discriminator to detect fake images
- Compare GAN architectures
- Provide prediction on user input images

---


### Preprocessing Steps

- Image resizing (64 × 64)
- Normalization
- Batch loading
- Data shuffling
- Train-test split

---

## GAN Architecture

### Generator

The generator network creates synthetic images from random noise.

Key Components:

- Transposed Convolution Layers
- Batch Normalization
- ReLU Activation
- Output Image Generation

---

### Discriminator

The discriminator network classifies images as real or fake.

Key Components:

- Convolution Layers
- LeakyReLU Activation
- Binary Classification
- Feature Extraction

---

## Training Configuration

| Parameter | Value |
|-----------|------|
| Epochs | 200 |
| Batch Size | 64 |
| Learning Rate | 0.0002 |
| Optimizer | Adam |
| Image Size | 64 × 64 |

---

## Training Workflow

1. Load dataset
2. Initialize generator and discriminator
3. Train discriminator on real and fake images
4. Train generator to fool discriminator
5. Record loss values
6. Save checkpoints
7. Evaluate model

---

## Visualization of Training Performance

### Generator Loss Curve

This plot shows how the generator loss changes during training.

![Generator Loss](generator_loss_plot.png)

Interpretation:

- Stable decrease indicates learning
- Sudden spikes indicate instability

---

### Discriminator Loss Curve

This plot shows how the discriminator loss evolves during training.

![Discriminator Loss](discriminator_loss_plot.png)

Interpretation:

- Balanced loss indicates healthy competition
- Very low loss may indicate overfitting

---

### Real vs Fake Score Separation

This plot shows how well the discriminator distinguishes between real and fake images.

![Real vs Fake Scores](real_vs_fake_scores_plot.png)

Interpretation:

- Higher separation indicates better performance
- Low separation indicates weak discrimination

---

## Model Comparison

### DCGAN

Advantages:

- Simple architecture
- Easy implementation
- Good baseline model

Limitations:

- Training instability
- Mode collapse
- Gradient vanishing

---

### WGAN

Advantages:

- Stable training
- Better gradient behavior
- Improved image quality
- Higher separation score

Limitations:

- Longer training time
- Higher computational cost

---

## Results Summary

| Model | Stability | Image Quality | Separation | Performance |
|------|-----------|--------------|-----------|------------|
| DCGAN | Moderate | Medium | Low | Weak |
| WGAN | High | Better | High | Best |

Conclusion:

WGAN performed better than DCGAN in terms of stability and discriminator performance.

---

## Model Saving

The trained model is saved using:
/home/CL502-25/Desktop/GAN_Dataset/results_dcgan/checkpoints/

---

## Inference System

The final system performs the following steps:

1. Load trained model
2. Accept user input image
3. Preprocess image
4. Run prediction
5. Display result

Output:
Real or Fake

---

## Applications

- Fake image detection
- Food authenticity verification
- Digital media validation
- AI research
- Computer vision education

---

## Future Improvements

- Use larger datasets
- Train for more epochs
- Deploy as web application
- Add real-time prediction
- Use advanced GAN architectures
- Calculate FID score

---

## Conclusion

This project successfully implemented and compared GAN architectures for generating and detecting food images. The results demonstrate that WGAN provides more stable training and better performance compared to DCGAN. The final system allows users to input an image and receive a prediction indicating whether the image is real or generated.

