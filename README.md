**Image Classification & Segmentation of Plant Seedlings**

This project implements various machine learning and deep learning models to classify and segment plant seedling images. It explores preprocessing, architecture design, training strategies, evaluation metrics, and fine-tuning to optimize performance.

------------------
**Data Preprocessing**

The dataset used in this project consists of 409 labeled RGB images representing different categories of plant seedlings. As a first step in the data preprocessing pipeline, random image samples are visualized to assess their quality, clarity, and label correctness. This helps identify any inconsistencies or potential anomalies in the dataset.
Next, all images are reshaped and resized to a standardized input dimension, ensuring compatibility with convolutional neural network (CNN) architectures that require fixed-size inputs.The images are then normalized by scaling pixel values from the original [0, 255] range to a [0, 1] range. This normalization step is essential for faster convergence during training, as it prevents large gradients and stabilizes the optimization process. Finally, the dataset is split into training, validation, and test sets, and the shape of each subset is inspected to confirm the correct data format and distribution.
This structured preprocessing ensures the dataset is clean, balanced, and ready for efficient model training.


