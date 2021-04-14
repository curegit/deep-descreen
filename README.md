# Anime Halftone Inversing

CNN solution for scanning anime prints

This project is still WIP, and its codes are extremely rough.
So they might not work properly.

## Aiming

When you digitize paper-printed anime arts, you have to smooth halftone dots (because of offset printing).
Usually, using filter-based methods (e.g. Gaussian blur) give undesired blurry results.
This project solves it by using CNN (Convolutional Neural Network) which convert dotted images into smooth image.

Here are the provisional result and comparison to the baseline (Gaussian blur).

![comparison](comparison.png)
