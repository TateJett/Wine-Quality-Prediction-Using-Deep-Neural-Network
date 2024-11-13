# Wine-Quality-Prediction-Using-Deep-Neural-Network
February 8, 2024

This project aims to predict the quality of white wine based on various chemical properties using a Deep Neural Network (DNN) implemented in PyTorch. The analysis explores relationships between features and applies a binary classification approach to assess wine quality as high or low.

## Project Overview

The main goals of this project are to:
- **Visualize and understand** the distributions of wine quality features.
- **Preprocess and prepare data** for training by scaling and splitting it into training and testing sets.
- **Build and train a DNN model** to predict wine quality based on chemical properties.
- **Evaluate model performance** and reflect on ethical implications in automated quality assessment.

## Dataset

The dataset includes various features that describe the chemical properties of white wine samples, including:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- **Target Variable**: Wine Quality (binary classification as high or low quality)

## Methodology

### 1. Data Visualization

- Created histograms for each feature to observe their distributions and understand their ranges.
- Used scatter matrices to explore potential relationships between key variables.
- Plotted a boxplot for the wine quality variable to visually assess the distribution of quality ratings.

![image](https://github.com/user-attachments/assets/3e78c34f-4674-4317-88ea-709f955bbeee)
![image](https://github.com/user-attachments/assets/008a608c-5b66-4bd6-935d-f605b17455cf)
![image](https://github.com/user-attachments/assets/1d87c933-ee2d-4f7b-80d7-c7e35d1505c0)

### 2. Data Preprocessing

- **Target Variable Separation**: The quality column was separated as the target variable for binary classification (high or low quality).
- **Standardization**: Features were standardized using `StandardScaler` to normalize the data for effective model training.
- **Train-Test Split**: The dataset was split into training (80%) and test (20%) sets.
- **Tensor Conversion**: The processed datasets were converted to PyTorch tensors for compatibility with the neural network.

### 3. Model Construction

- Built a custom PyTorch neural network class (`PyTorch_NN`) by extending `nn.Module`.
- **Model Architecture**:
  - Input layer with 11 neurons, corresponding to the 11 features.
  - Two output neurons for binary classification (high or low quality).
  - Intermediate hidden layers with varying neuron counts and ReLU activation functions.
- **Loss Function**: Chose CrossEntropyLoss for binary classification.
- **Optimizer**: Used the Adam optimizer with a learning rate of 0.001, based on prior experience and its adaptability for deep learning tasks.

### 4. Model Training

- **Training Process**: The network was trained for 1000 epochs.
- **Loss Tracking**: Printed the loss at every 100th epoch to monitor learning progress.
- **Backpropagation**: Utilized PyTorch’s backpropagation and optimization methods (zero_grad, loss.backward, and optimizer.step) to update weights after each epoch.

### 5. Model Evaluation

- The model was tested on the reserved test set, achieving an accuracy of **76.53%**.
- This accuracy suggests the model performs reasonably well in classifying wine quality, although improvements could be made with further tuning or additional data.

## Results

- **Accuracy**: The model achieved an accuracy of 76.53% on the test set.
- **Usefulness**: This performance indicates that the model is potentially useful for automated wine quality assessment, though further refinement is necessary for real-world applications.
- **Ethical Considerations**: Automated quality assessment should be complemented by human expertise to avoid overlooking nuanced factors. There is also the possibility of built-in bias within the dataset, which could impact fairness and accuracy.

## Discussion and Future Work

- **Further Exploration**: The project could be expanded by experimenting with different neural network architectures, hyperparameters, and additional feature engineering.
- **Real-World Applications**: While the model shows promise, it should be used alongside traditional methods and expert knowledge to ensure reliable results.
- **Ethics**: Ethical implications of relying solely on algorithmic predictions in quality assessment should be taken into account, as algorithmic bias or oversights could have unintended impacts.

## Installation and Usage

### Prerequisites

- Python 3.x
- Required libraries: `pandas`, `numpy`, `scipy`, `matplotlib`, `sklearn`, `torch`

### Running the Project

1. **Download the Notebook**:
   - Download `AML_project_1.ipynb` from this repository.

2. **Install Dependencies**:
   - Open a terminal and run:
   ```bash
   pip install pandas numpy scipy matplotlib scikit-learn torch
3. **Open and Execute the Notebook**:
   - Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab to follow the full workflow, including data visualization, model construction, training, and evaluation.

## Visualizations

This project includes several key visualizations:

1. **Feature Histograms**:
   - Visual representation of the distribution of each feature.
![image](https://github.com/user-attachments/assets/3e78c34f-4674-4317-88ea-709f955bbeee)

2. **Scatter Matrices**:
   - Shows relationships between key variables to explore potential correlations.
![image](https://github.com/user-attachments/assets/008a608c-5b66-4bd6-935d-f605b17455cf)

3. **Wine Quality Boxplot**:
   - Displays the distribution of wine quality ratings.
![image](https://github.com/user-attachments/assets/1d87c933-ee2d-4f7b-80d7-c7e35d1505c0)
