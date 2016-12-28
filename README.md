# libskeletal
Implementation of a human pose estimation algorithm (modified Shotton et al)

## Algorithm
`libskeletal` is based on the offset-joint regression (OJR) algorithm presented in "Efficient Pose Estimation from Single Depth Images" et al. It is implemented in Python with opencv (for image processing), numpy, and scikit-learn (for random forests). It can currently run in real-time at low-resolutions on a single thread; the algorithm itself is highly parallelizable and can be implemented on GPUs, as is done in the Xbox implementation of the same algorithm. Small modifications and simplifications have been made with respect to the exact choice of features and post-processing methods. 

## Choice of features
The original algorithm was designed for use with depth images, with optional support for silhouettes extracted from a monocular video camera. libskeletal is particularly focused on monocular video and uses a novel feature incorporating a skin color detector. The skin color detector itself is from "A comparative assessment of three approaches to pixel level human skin detector" by Brand and Mason, chosen for simplicity and performance. In total, this enables libskeletal to work effectively on monocular video directly.
