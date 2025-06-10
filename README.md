# Pneumonia Detection from Chest X‑Rays

This repository contains a convolutional neural network (CNN) pipeline to classify chest X‑ray images as **Pneumonia** or **Normal**. The end‑to‑end workflow covers:

1. **Data Loading & Preprocessing**  
   - Grayscale images resized to 150×150 px  
   - Train / validation / test split from separate folders  
   - Class‑imbalance handling via computed class weights  
   - Real‑time data augmentation (rotation, zoom, shifts, flips)

2. **Model Definition**  
   - **5 Conv‑blocks** with increasing filters (32→64→64→128→256):  
     - `Conv2D` → `BatchNormalization` → (optional `Dropout`) → `MaxPool2D`  
   - **Dense head**:  
     - Flatten → Dense(128, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)  
   - Compiled with **RMSprop** optimizer and **binary_crossentropy** loss  

3. **Training**  
   - 12 epochs with **ReduceLROnPlateau** callback to adapt learning rate  
   - Validation monitored on augmented data  
   - Early improvements in accuracy up to ~96% (train) and ~87% (val)

4. **Evaluation & Visualization**  
   - Final **test accuracy**: ~89.4%  
   - Precision / recall / F1 for each class  
   - Confusion matrix heatmap  
   - Sample grids of correctly vs. incorrectly classified X‑rays  
   - Training curves (accuracy & loss)

---

## Usage

1. Clone the repo and enter its folder:  
   ```bash
   git clone https://github.com/your‑username/pneumonia‑detector.git
   cd pneumonia‑detector
