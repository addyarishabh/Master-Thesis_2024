# Enhancing Diabetic Retinopathy Detection Using a ConvNeXt-KAN Deep Learning Model 


![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/title%20image2.jpg?raw=true)

### Abstract:

Diabetic Retinopathy (DR), a leading cause of vision impairment, results from prolonged hyperglycemia damaging retinal blood vessels. Early detection is critical to prevent severe vision loss. This study proposes a hybrid deep learning model combining ConvNeXt for automated feature extraction and a Kolmogorov-Arnold Network (KAN) for classification, aiming to improve diagnostic accuracy, interpretability, and clinical reliability in DR severity detection.
A dataset of retinal fundus images, categorized into five DR severity levels, was used. ConvNeXt extracted deep features, reducing manual feature engineering, while the KAN-based classifier employed functional decomposition for transparent decision-making. The model was trained using cross-entropy loss and optimized with the Adam optimizer.
Results showed the ConvNeXt-KAN model achieved 96.75% classification accuracy, outperforming traditional deep learning methods. It also demonstrated balanced performance with an average sensitivity (recall) of 96.8% and specificity of 99.19% across DR severity levels. AUC-ROC analysis confirmed robustness, with AUC scores exceeding 0.99 for all classes. The inclusion of KAN enhanced interpretability and decision transparency, crucial for clinical adoption.
Globally, DR prevalence among individuals with diabetes is significant, with projections estimating 160.5 million affected by 2045, including 44.82 million in vision-threatening stages. In India, DR prevalence is 16.9%, with 3.6% at sight-threatening stages, highlighting the urgent need for effective screening. This study demonstrates that integrating deep feature extraction with KAN-based classification improves both accuracy and interpretability in DR detection, offering a promising tool for early diagnosis and management. Such advancements are vital to reduce the global and national burden of DR-related vision loss.
Keywords: Diabetic Retinopathy (DR), Deep Learning, ConvNeXt, Kolmogorov-Arnold Network (KAN), Severity Levels. 


![login](https://github.com/addyarishabh/Master-Thesis_2024/blob/main/icon/Title%20image.png?raw=true)


### Problem Statement:

* Despite the advancements in deep learning for DR detection, several 
challenges remain: 
Limited interpretability – Traditional CNN-based models function as "black 
boxes," making clinical validation difficult. 
High computational cost – Pre-trained models such as ViTs require large
scale data and extensive computation. 
Class imbalance in medical datasets – Severe DR cases are significantly 
underrepresented, leading to poor classification performance in minority 
classes. 
Variability in retinal images – Differences in illumination, contrast, and 
imaging devices affect model robustness. 
* To overcome these issues, this study integrates Kolmogorov-Arnold 
Networks (KAN) into ConvNeXt, replacing the traditional MLP (Multilayer 
Perceptron) head with learnable B-spline activation functions. This 
approach enhances the model’s interpretability, reduces parameter overhead, 
and improves generalization. Additionally, the proposed model is deployed 
using Flask and CSS, making it accessible for real-world applications.









