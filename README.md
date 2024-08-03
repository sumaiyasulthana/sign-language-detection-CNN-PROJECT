# sign-language-detection-CNN-PROJECT

**LinkedIn :** https://www.linkedin.com/in/sumaiya-sulthana-906876137

**Output Video** Link:https://drive.google.com/file/d/1s8ur3e-6bNpeHfHMamODzunty5OcDzrL/view?usp=drive_link

**Objective:**
        The primary goal of this project is to create a Deep learning model capable of identifying Sign Language letters from hand gesture images. This involves building, training, and evaluating both Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN), and leveraging transfer learning with VGG16.

**Use Cases:**
        Sign language detection using Convolutional Neural Networks (CNN) can revolutionize communication for individuals who are deaf or hard of hearing. By creating a system that translates sign language into text or spoken words in real-time, businesses can significantly enhance accessibility and inclusivity. This technology can be integrated into various applications such as customer service, healthcare, education, and more.
        
**Dataset Explanation:**

   **Source:** The dataset consists of hand gesture images representing SL letters.
   
   **Classes:** 24 classes (A-I, K-Y), excluding J and Z due to their motion requirements.Each training and test case
     represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture 
     motions).
             
   **Format:** Similar to the MNIST dataset, with 28x28 grayscale images.
   
   **Size of dataset:** Training set: 27,455 images,
                        Test set: 7,172 images
                        
**Steps involved:**    

**Data Preprocessing:**

1.**Normalization:** Pixel values are normalized to the range [0, 1].by dividing 255

2.**Reshaping:** Images are reshaped to (28, 28, 1) for ANN/CNN models and resized to (128, 128, 3) **for VGG16**.
                And changed grayscale channel to to RGB.
               
3.**One-Hot Encoding:** Labels are one-hot encoded for training.(25 Labels)
       
**Model Building:**

  1.**ANN Model:**
**Architecture:** A simple fully connected neural network with:

**Input layer:** 784 neurons (28x28 flattened image)

**Hidden layers:** Dense layers with ReLU activation

**Output layer:** 25 neurons (softmax activation)

**Training:** Used Adam optimizer and categorical cross-entropy loss.
       
  2.**CNN Model:**
**Architecture:** A convolutional neural network with:
                  Convolutional layers,
                  Max-pooling layers,
                  Dense layers
                  
**Output layer:** 25 neurons (softmax activation)

**Training:** Used Adam optimizer and categorical cross-entropy loss.
       
 3.**Transfer Learning with VGG16:**
 **Architecture:**
 **Base model:** VGG16 (pre-trained on ImageNet, without the top layers)
 
 **Added layers:** Batch normalization, global average pooling, dense layers, dropout for regularization.
 
 **Output layer:** 25 neurons (softmax activation)
 
 **Freezing Layers:** The base VGG16 layers were frozen to prevent their weights from being updated during training.
 
 **Training:** Used Adam optimizer and categorical cross-entropy loss.
         
**Model Training:**
**Epochs:** Adjusted based on performance and early stopping.

**Early Stopping:** Monitored validation loss with a patience of 10 epochs and a min_delta of 0.001.
        
**Model Evaluation:**
**Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

**Visualization:** Display of true vs. predicted labels on test images.

**FEATURES of my App:** 

**Testing and Prediction:**
**Interactive Testing:** A function allows users to input an index (0-23) and view the corresponding letter's image and 
 prediction.
 
**Example:** The user provides an index, and the function displays the true label, predicted label, and the corresponding image.

**REQUIRED LIBRARIES**:

1.Scikit-learn

2.TensorFlow---Keras

3.Opencv (computer vision)

4.numpy

5.pandas

6.seaborn

7.matplotlib.pyplot

**Domain:** Deep Learning.

 








       
