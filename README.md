**Project Overview**

This project involves building a deep learning pipeline for detecting faces and classifying whether they are wearing a mask. It leverages pre-trained models and facial bounding box annotations to train and evaluate a U-Net-like segmentation model tailored for face region recognition.

----------------
**Data Preprocessing**

In the data preprocessing stage, each facial image was resized to a uniform dimension of 224×224 using OpenCV’s cv2.resize, followed by normalization using the preprocess_input function from MobileNet to ensure pixel values were appropriately scaled for deep learning models. Binary masks were generated from the provided facial bounding box annotations, where each bounding box was mapped to its corresponding spatial location and marked within a zero-initialized mask tensor. To maintain consistency across samples, grayscale images were converted to RGB format, ensuring a standardized three-channel input. As a result, the final input tensor shape for the model was (num_samples, 224, 224, 3), and the corresponding mask tensor shape was (num_samples, 224, 224).

--------------------
**Model Architecture**

The segmentation model employed a U-Net–like structure with a MobileNet backbone for feature extraction and a custom decoder for spatial reconstruction. The encoder utilized pretrained MobileNet convolutional layers to efficiently capture hierarchical facial features. The decoder consisted of upsampling blocks, skip connections, and convolutional layers to refine predictions and generate segmentation masks. The model accepted 3-channel RGB images as input and produced binary masks indicating the presence of facial regions associated with mask detection.

-----------------
**Methodology**

The pipeline was designed for semantic segmentation using facial bounding box supervision. For each input image, a corresponding binary mask was generated using annotated bounding boxes. The model was trained to segment these regions, learning to associate spatial cues with mask presence. The approach combined transfer learning (MobileNet) with custom decoding layers to strike a balance between accuracy and computational efficiency. This methodology supports end-to-end training using paired image and mask inputs, enabling pixel-wise prediction.

-------------------
**Model Training**

The model was trained using the Binary Cross-Entropy (BCE) loss function to optimize pixel-level classification, with optional use of the Dice loss to improve spatial overlap performance. Training was conducted over multiple epochs, with batch processing and early stopping based on validation performance. The training pipeline included shuffling, batching, and on-the-fly preprocessing. Validation metrics were monitored at each epoch to prevent overfitting and guide hyperparameter refinement.

----------------------
**Hyperparameter Tuning**

Hyperparameter optimization involved tuning the learning rate, optimizer type (e.g., Adam, SGD), dropout rates, and the depth of decoder layers. Dropout layers were introduced after convolutional blocks to regularize the model and improve generalization. The learning rate was adjusted iteratively based on convergence behavior, and training duration was controlled to avoid overfitting. Image size was fixed at 224×224 to match the backbone's expected input dimensions.

--------------------
**Model Evaluation**

Model performance was evaluated using pixel-level classification metrics such as accuracy, precision, recall, and F1 score, with a particular focus on Intersection over Union (IoU) and Dice coefficient to assess segmentation quality.I have achieved a Dice coefficient around 0.85, an IoU score near 0.75, precision and recall values in the range of 0.86–0.88, and an F1 score of approximately 0.87. 
