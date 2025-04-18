# KAN-Augmented ConvNeXt Model for Robust Multi-Class Diabetic Retinopathy Classification

![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/title%20image2.jpg?raw=true)

#### Submitted by,
Rishabh Kumar Addya (19458122003)

#### Degree Programme: 
Master of Sceince 


#### Batch: 
2022-2024

#### Year:
2023-2024

#### Under the Supervision of:
Prof. Mithun Mazumdar (HOD, Data Analytics Department)

#### Submitted to, 
Data Analytics

#### at
Institute of Management Study


![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/logo.png?raw=true)


### Abstract:

Diabetic Retinopathy (DR), a leading cause of vision impairment, results from prolonged hyperglycemia damaging retinal blood vessels. Early detection is critical to prevent severe vision loss. This study proposes a hybrid deep learning model combining ConvNeXt for automated feature extraction and a Kolmogorov-Arnold Network (KAN) for classification, aiming to improve diagnostic accuracy, interpretability, and clinical reliability in DR severity detection.
A dataset of retinal fundus images, categorized into five DR severity levels, was used. ConvNeXt extracted deep features, reducing manual feature engineering, while the KAN-based classifier employed functional decomposition for transparent decision-making. The model was trained using cross-entropy loss and optimized with the Adam optimizer.
Results showed the ConvNeXt-KAN model achieved 96.75% classification accuracy, outperforming traditional deep learning methods. It also demonstrated balanced performance with an average sensitivity (recall) of 96.8% and specificity of 99.19% across DR severity levels. AUC-ROC analysis confirmed robustness, with AUC scores exceeding 0.99 for all classes. The inclusion of KAN enhanced interpretability and decision transparency, crucial for clinical adoption.
Globally, DR prevalence among individuals with diabetes is significant, with projections estimating 160.5 million affected by 2045, including 44.82 million in vision-threatening stages. In India, DR prevalence is 16.9%, with 3.6% at sight-threatening stages, highlighting the urgent need for effective screening. This study demonstrates that integrating deep feature extraction with KAN-based classification improves both accuracy and interpretability in DR detection, offering a promising tool for early diagnosis and management. Such advancements are vital to reduce the global and national burden of DR-related vision loss.
Keywords: Diabetic Retinopathy (DR), Deep Learning, ConvNeXt, Kolmogorov-Arnold Network (KAN), Severity Levels. 


![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/Title%20image.png?raw=true)


### Problem Statement:

1) Despite the advancements in deep learning for DR detection, several 
challenges remain: 
- Limited interpretability – Traditional CNN-based models function as "black 
boxes," making clinical validation difficult. 
- High computational cost – Pre-trained models such as ViTs require large
scale data and extensive computation. 
- Class imbalance in medical datasets – Severe DR cases are significantly 
underrepresented, leading to poor classification performance in minority 
classes. 
- Variability in retinal images – Differences in illumination, contrast, and 
imaging devices affect model robustness. 
2) To overcome these issues, this study integrates Kolmogorov-Arnold 
Networks (KAN) into ConvNeXt, replacing the traditional MLP (Multilayer 
Perceptron) head with learnable B-spline activation functions. This 
approach enhances the model’s interpretability, reduces parameter overhead, 
and improves generalization. Additionally, the proposed model is deployed 
using Flask and CSS, making it accessible for real-world applications.


### Aim:

To develop and evaluate a ConvNeXt-KAN hybrid model for the accurate and 
interpretable classification of diabetic retinopathy severity. 


### Research Questions:

a) How does the ConvNeXt-KAN model improve DR classification accuracy 
compared to MobileNetV2, EfficientNet-B0, and ViT? 

b) Can KAN-based classification enhance model interpretability over traditional 
CNN architectures? 

c) How does the model perform in terms of sensitivity and specificity, especially 
for proliferative DR cases? 

d) What are the benefits of replacing MLP heads with KAN-based functional 
decomposition in deep learning models? 

e) How can the proposed system be deployed effectively for real-time DR 
screening in clinical settings?


### Literature Review:

Diabetic Retinopathy (DR) is one of the leading causes of blindness worldwide, primarily 
caused by prolonged hyperglycemia leading to damage in retinal blood vessels. Several deep 
learning-based approaches have been proposed for the automated detection and classification 
of DR. Traditional machine learning methods, such as Support Vector Machines (SVM) and 
Random Forests, were used in earlier studies but required handcrafted feature extraction, 
limiting their generalization ability. 
With the advent of Convolutional Neural Networks (CNNs), deep learning models such as 
AlexNet, VGG16, ResNet, and EfficientNet have significantly improved DR classification by 
learning hierarchical feature representations from retinal fundus images. The EyePACS 
dataset and the APTOS dataset are commonly used for training and validating such models. 
Studies have reported high classification accuracy using pre-trained models like InceptionV3 
(92.1%), ResNet50 (91.3%), and EfficientNet-B0 (93.5%). However, most CNN-based 
models face interpretability challenges and require large amounts of labeled data for effective 
training. Recent advancements have explored Vision Transformers (ViTs) for DR 
classification due to their ability to model global dependencies in images. However, ViTs 
often require high computational resources and large-scale pre-training on medical image 
datasets, making them less accessible for real-world clinical applications. 
Thus, while existing deep learning approaches have shown promise in DR detection, they still 
suffer from computational inefficiencies, lack of interpretability, and suboptimal 
generalization to real-world clinical scenarios. This study aims to address these limitations by 
integrating ConvNeXt for feature extraction and Kolmogorov-Arnold Networks (KAN) for 
classification, leveraging their unique advantages. 

### Multi-Layer Perceptron (MLP) vs Kolmogorov-Arnold Network (KAN)

![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/MLP%20vs%20KAN.png?raw=true)


### End to End Pipeline for ConvNeXt-KAN-Based Diabetic Retinopathy Detection

![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/pipeline.png?raw=true)

### Tools Used:

![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/tools.jpeg?raw=true)


### Dataset Description:

The dataset consists of retinal fundus images used for detecting diabetic retinopathy. The 
original dataset was sourced from APTOS 2019 Blindness Detection and contained images of 
different severity levels of diabetic retinopathy. Each image was resized to 224×224 pixels to match the input size of pre-trained deep learning 
models. 
The images were categorized into five classes based on the severity of diabetic retinopathy:

* Label 0 -> Healthy Retina - Original Count (1805)
* Label 1 -> Mild DR - Original Count (370)
* Label 2 -> Moderate DR - Original Count (999)
* Label 3 -> Severe DR - Original Count (193)
* Label 4 -> Proliferative DR - Original Count (295)

To ensure a balanced dataset, the number of images per class was equalized to 400 per class, 
leading to a total of 2000 images used for training. 

### Train-Test Split:

* 80% Training Set (1600 images)
* 20% Test Set (400 images)


### ConvNeXt-KAN model summary:

![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/model.png?raw=true)

### Model Training Loss and Training Accuracy Graph:

![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/loss%20vs%20accuracy.png?raw=true)


### Model Evaluation:

![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/metrics.png?raw=true)

### Confusion Matrix:

![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/c12.png?raw=true)

### AUC-ROC Curve:


![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/f12.png?raw=true)


### Local Application Interface

![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/interface.png?raw=true)


### Results:

![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/result.png?raw=true)


### Limitations:

Despite achieving promising results, several limitations were identified in this study:
1. Dataset Limitations
-	The dataset used in this study contains 2,000 images, which, while balanced, is not large enough for a production-level system.
-	A larger, more diverse dataset with real-world clinical images would enhance generalization.
2. Computational Complexity
-	ConvNeXt, being a highly parameterized deep learning model, requires substantial computational resources.
-	Training on edge devices or lower-end GPUs may not be feasible without model pruning or quantization.
3.	Lack of Lesion Localization
-	The current model classifies the entire image into a DR category but does not localize specific lesions (such as microaneurysms or hemorrhages).
-	Localization techniques, such as Grad-CAM or attention mechanisms, could enhance interpretability.


### References:

[1]. Neural Computing and Applications, "Automated Detecting and Severity 
Grading of Diabetic Retinopathy Using Transfer Learning and Attention 
Mechanism," 2023. DOI:  
https://doi.org/10.1007/s00521-023-09001-1

[2]. arXiv preprint arXiv:2308.09945, "Dual Branch Deep Learning Network for 
Detection and Stage Grading of Diabetic Retinopathy," 2023. Available at: 
https://arxiv.org/abs/2308.09945

[3]. arXiv preprint arXiv:2405.01734, "Diabetic Retinopathy Detection Using 
Quantum Transfer Learning," 2024. Available at: 
https://arxiv.org/abs/2405.01734

[4]. arXiv preprint arXiv:2301.00973, "Detecting Severity of Diabetic Retinopathy 
from Fundus Images: A Transformer Network-based Review," 2023. Available at: 
https://arxiv.org/abs/2301.00973

[5]. arXiv preprint arXiv:2004.06334, "Automated Diabetic Retinopathy Grading 
Using Deep Convolutional Neural Network," 2020. Available at: 
https://arxiv.org/abs/2004.06334

[6]. Sensors, "Using Deep Learning Architectures for Detection and Classification 
of Diabetic Retinopathy," 2023. DOI: 
https://doi.org/10.3390/s23125726

[7]. Journal of Electrical Systems, "Diabetic Retinopathy Detection Using Deep 
Learning," 2023. Available at: 
https://journal.esrgroups.org/jes/article/view/687

[8]. Multimedia Tools and Applications, "Diabetic Retinopathy Prediction Based 
on Deep Learning and Deformable Registration," 2022. DOI: 
https://doi.org/10.1007/s11042-022-12968-z

[9]. Diagnostics, "Enhancement of Diabetic Retinopathy Prognostication Using 
Deep Learning, CLAHE, and ESRGAN," 2023. DOI: 
https://doi.org/10.3390/diagnostics13122001

[10]. Digital Health, "Enhancing Diabetic Retinopathy Classification Using Deep 
Learning," 2023. DOI: 
https://doi.org/10.1177/20552076231100758 

[11] Dataset link: 
https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy
224x224-2019-data/ 
[12] Project Github Link: 
https://github.com/addyarishabh/Master-Thesis_2024.git





