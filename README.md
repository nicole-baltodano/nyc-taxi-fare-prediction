# NYC Taxi Fare Prediction

This repository contains the code and workflow for building a **Neural Network-based model** to predict taxi fares in New York City based on a Kaggle dataset.

---

## **Dataset Overview**

The dataset is sourced from [Kaggle's New York City Taxi Fare Prediction competition](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction). It includes information about taxi rides such as the pickup and drop-off coordinates, date and time, passenger count, and the fare amount.

### **Preview of the Dataset**

| pickup_datetime       | pickup_longitude | pickup_latitude | dropoff_longitude | dropoff_latitude | passenger_count | fare_amount |
|-----------------------|------------------|-----------------|-------------------|------------------|-----------------|-------------|
| 2013-07-06 17:18:00  | -73.950655       | 40.783282       | -73.984365        | 40.769802        | 1               | 12.50       |
| 2013-07-06 17:19:00  | -73.948655       | 40.784282       | -73.982365        | 40.768802        | 2               | 8.70        |
| 2013-07-06 17:20:00  | -73.949655       | 40.785282       | -73.980365        | 40.766802        | 1               | 5.30        |

---

## **Model Architecture**

The model used for this project is a fully connected neural network (Multi-Layer Perceptron). The architecture includes:

- **Input Layer**:
  - Accepts features like pickup and drop-off coordinates, pickup time, and passenger count.
- **Hidden Layers**:
  - Two fully connected layers with 100 and 50 units, respectively, each using ReLU activation.
  - Batch Normalization for stabilizing training and improving convergence.
  - Dropout (10%) for regularization to prevent overfitting.
- **Output Layer**:
  - A single unit with a linear activation function to predict the fare amount.

## **Model Performance**

The Root Mean Square Error (RMSE) of the model on the test set is **3.99**. Below is a visualization of the predicted versus actual fare amounts:

![image](https://github.com/user-attachments/assets/74376c4b-9b5c-436d-9f15-631474984496)

- The red dashed line represents perfect predictions where `Predicted = Actual`.
- Most of the predictions align closely with the actual values, indicating good model performance.

## **Installation**

`git clone https://github.com/nicole-baltodano/nyc-taxi-fare-prediction.git`
`cd nyc-taxi-fare-prediction`
`pip install -r requirements.txt`
