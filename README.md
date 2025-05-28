### Mitochondria Segmentation using U-Net

This project implements a U-Net model for segmenting mitochondria in electron microscopy images. The model is built using TensorFlow and Keras.

#### Project Overview

The primary goal is to accurately identify and delineate mitochondria regions within given images. The notebook covers the following key steps:
1.  **Data Loading and Preprocessing:**
    *   Loading `.tif` image and corresponding ground truth mask files.
    *   Normalizing image pixel values.
    *   Binarizing mask values.
    *   Reshaping data to be compatible with the U-Net model input.
2.  **Dataset Splitting:** Dividing the dataset into training and testing sets.
3.  **Model Definition:** Implementing the U-Net architecture, a convolutional neural network designed for biomedical image segmentation.
    *   The architecture consists of an encoder path (to capture context) and a decoder path (to enable precise localization).
    *   It utilizes convolutional layers, max pooling, dropout for regularization, and up-sampling (Conv2DTranspose) layers.
4.  **Model Compilation:** Configuring the model with an Adam optimizer, binary cross-entropy loss function, and metrics like accuracy and Dice coefficient.
5.  **Model Training:** Training the U-Net model on the prepared training dataset.
6.  **Evaluation:**
    *   Plotting training history (loss, accuracy, Dice coefficient) to monitor performance.
    *   Evaluating the trained model on the test set.
7.  **Prediction and Visualization:** Making predictions on test images and visualizing the original image, its ground truth mask, and the model's predicted mask side-by-side for qualitative assessment.

#### Dependencies

The project relies on the following Python libraries:
*   `os`
*   `numpy`
*   `cv2` (OpenCV-Python)
*   `sklearn` (specifically `model_selection.train_test_split`)
*   `tensorflow`
*   `keras` (as part of TensorFlow)
*   `matplotlib` (for plotting)

#### Dataset

The dataset is expected to be structured as follows:
*   Images: `256x256_mitochondria/images/` (containing `.tif` image files)
*   Ground Truth Masks: `256x256_mitochondria/groundtruths/` (containing corresponding `.tif` mask files)

The images are grayscale and 256x256 pixels in dimension. Masks are binarized during preprocessing.

#### Workflow

1.  **Setup:**
    *   Ensure all dependencies are installed.
    *   Place the dataset in the specified directory structure.
2.  **Execution:**
    *   Run the Jupyter Notebook `280201075_HW3.ipynb`.
    *   The notebook will:
        *   Load and preprocess the data.
        *   Define and compile the U-Net model.
        *   Train the model.
        *   Display training progress and evaluation metrics.
        *   Show visual results of segmentation on test images.

#### Model Architecture: U-Net

The U-Net model implemented includes:
*   **Encoder:**
    *   Multiple blocks of two 3x3 Convolutional layers (ReLU activation) followed by a 2x2 Max Pooling layer.
    *   Dropout layers are included for regularization.
*   **Bottleneck:**
    *   Two 3x3 Convolutional layers with ReLU activation.
*   **Decoder:**
    *   Multiple blocks of a 2x2 UpSampling (Conv2DTranspose) layer, concatenation with the corresponding feature map from the encoder path (skip connection), and two 3x3 Convolutional layers (ReLU activation).
*   **Output Layer:**
    *   A 1x1 Convolutional layer with a sigmoid activation function to produce a probability map for the segmentation.

#### Custom Metrics
*   **Dice Coefficient:** Used as a metric during training and evaluation to measure the overlap between the predicted segmentation and the ground truth.
    
    \[
    \text{Dice Coefficient} = \frac{2 \times |X \cap Y|}{|X| + |Y|}
    \]
    
    Where X is the predicted set of pixels and Y is the ground truth set of pixels.

#### How to Use

1.  Clone or download the repository/notebook.
2.  Make sure you have Python and the listed dependencies installed.
3.  Organize your dataset (`256x256_mitochondria`) in the same directory as the notebook, or update the paths in the notebook accordingly.
4.  Open and run the `280201075_HW3.ipynb` notebook in a Jupyter environment.

This README provides a good overview of your project. You can copy and paste this into a `readme.md` file in your project directory.