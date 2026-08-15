## Executive Summary

This project investigates **multi-class scene classification** on the Intel Image Classification dataset, covering six classes: *buildings, forest, glacier, mountain, sea,* and *street*. The objective was to evaluate the effectiveness of both **handcrafted visual features with classical machine learning** and **learned representations using convolutional neural networks (CNNs)**.

The initial classical pipeline combined **HOG and LBP feature extraction, feature scaling, PCA-based dimensionality reduction, and supervised classifiers including SVM and Logistic Regression**. This provided a structured baseline while highlighting the limitations of handcrafted features for visually similar scenes, particularly *glacier, mountain,* and *sea*. 

A CNN was subsequently developed in **PyTorch** to learn image representations directly from pixel data. A controlled experiment compared a **baseline CNN** with a **ColorJitter-augmented CNN**, keeping the train/test methodology and optimization settings consistent. The baseline achieved **84.17% test accuracy**, while ColorJitter increased accuracy to **86.47%**, a **2.30 percentage-point improvement**. Macro F1 increased correspondingly from approximately **84.12% to 86.49%**, indicating that the improvement was not limited to the largest classes.

Model evaluation was extended beyond aggregate metrics using **confusion matrices and Grad-CAM visualisation**. Grad-CAM was used to inspect the spatial regions driving CNN predictions, providing an interpretability check on whether the learned representations were responding to meaningful scene content rather than relying predominantly on irrelevant image regions.

Overall, the project demonstrates a progression from **engineered visual representations → dimensionality reduction → classical classification → learned CNN representations → controlled augmentation → model interpretability**, providing both quantitative and qualitative evidence for the advantages of learned representations in scene classification. The repository contains modular feature-extraction, model-training and evaluation components supporting this experimental workflow. 
