# Brain Tumor Classification with Deep Learning and Fastai

In this project, a brain tumor dataset including 3 types of tumors that are glioma, meningioma, pituitary and normal brain MR images with 256x256 size was chosen from Kaggle to observe the transfer learning process of a deep learning model. 
Of course, deciding the topic to work on it and choosing a dataset appropriate to the aim is not enough to achieve a well performance model. The examining and processing the dataset is an indispensable part of starting to get the desired model.
Here, my decision making map to get the last version of the model:

#### 1. Data Loading
#### 2. Data Splitting
#### 3. Data Block Creation
#### 4. Batch Observation
#### 5. Model Training Process with Transfer Learning
##### 5.1 Benchmark Creation
##### 5.2 Fine Tune with Freezing Layers
##### 5.3 Learning Rate Finder
##### 5.4 Fine Tuning with Decided Learning Rate
##### 5.5 Fit One Cycle with Unfreezing Frozen Layers
##### 5.6 Discriminative Learning Rates
#### 6. Finding Best Threshold and Testing
#### 7. Exporting The Trained Model
#### 8. ResNet18 and Resnet50 Comparison
#### 9. Gradio Application Example Predictions

I explained the idea of training a deep learning vision model which is ResNet versions in this case with various transfer learning techniques. How I decided to take those steps like epoch number, learning rate decisions? What was my benchmark to process the next step? I explained in depth the perspective of transfer learning of a vision model in my medium blog post. You may observe the process in a detailed way from blog and experience the model using Hugging Face Spaces link.

Hugging Face Spaces: https://busra-emir-brain-tumor-classification

Medium Blog Post: 

Kaggle Dataset: https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256


Example Outputs:
![image](https://github.com/user-attachments/assets/b32b76fe-8c63-42f2-937d-ac8f7645ed00)
![image](https://github.com/user-attachments/assets/c5b56324-4609-43c2-91e6-cc05ca7913ed)
![image](https://github.com/user-attachments/assets/131a166c-8189-4b22-87d2-a6bfb43b95ed)

