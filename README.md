# Integrated-AI-Cancer-Diagnosis
This project focuses on building a deep learning-based system for detecting and classifying different types of cancer from medical images. The system utilizes Convolutional Neural Networks (CNNs) to identify and categorize tumors in brain MRI scans and detect breast and lung cancers with high accuracy. Additionally, the project integrates a locally deployed web application to provide an accessible and interactive interface for users, allowing them to upload images and view predictions seamlessly.

# Key Features

Multi-Cancer Detection:
  Detects and classifies:
    Brain tumors into three categories: Glioma, Meningioma, and Pituitary tumors.
    Breast and lung cancers as "Cancer" or "No Cancer."

Deep Learning Approach:
  Utilizes a CNN model with a pre-trained base and custom layers to extract essential features and perform classification tasks with high accuracy.

Dataset and Preprocessing:
  Curated datasets with labeled medical images are used for model training and testing.
  Preprocessing includes resizing and normalization to enhance model performance and generalization.

Locally Deployed Web Application:
  A Flask-based web application allows users to interact with the system.
  Users can upload medical images, and the application provides predictions with visual results, including classifications.

User-Friendly Interface:
  Results are presented in a clear and concise format, providing a breakdown of the predictions, including the detected cancer type or confirmation of its absence.
