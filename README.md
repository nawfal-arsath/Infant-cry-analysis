# Don't Cry - AI-Powered Infant Cry Analysis Web App  

## 📌 Overview  
**Don't Cry** is an AI-powered web application that helps parents and caregivers understand why their baby is crying. The system analyzes infant cry audio recordings and provides predictions using a deep learning model.  

## 🔥 Features  
- 🎙️ Upload baby cry audio for analysis  
- 🤖 AI-powered predictions on cry reasons  
- 🌐 User-friendly web interface  
- ⚡ Fast and accurate results  

## 🚀 Tech Stack  
- **Frontend:** React.js, Tailwind CSS  
- **Backend:** FastAPI / Flask  
- **AI Model:** TensorFlow / PyTorch  
- **Deployment:** Vercel / AWS / Firebase  

## 🏋️ Model Training Process  

### 📊 Dataset  
- The dataset consists of infant cry recordings labeled with different reasons (e.g., hunger, discomfort, pain, sleepiness).  
- Preprocessing includes noise reduction, feature extraction, and data augmentation.  

### 🎛️ Preprocessing Steps  
1. **Audio Conversion:** Convert audio files to a uniform format (e.g., 16kHz, mono).  
2. **Noise Reduction:** Apply filters to remove background noise.  
3. **Feature Extraction:** Extract Mel-Frequency Cepstral Coefficients (MFCCs) and spectrograms.  
4. **Augmentation:** Add variations (pitch shift, speed changes) to improve model robustness.  

### 🤖 Model Architecture  
- **Base Model:** CNN-BiLSTM hybrid for capturing temporal dependencies in audio.  
- **Feature Extraction:** Using MFCCs and Mel-spectrograms.  
- **Classifier:** Fully connected layers with softmax activation for multi-class classification.  

### 🏋️ Training Details  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Batch Size:** 32  
- **Epochs:** 50  
- **Validation Split:** 20%  

### 📈 Performance & Evaluation  
- **Accuracy:** Achieved >85% validation accuracy.  
- **Metrics:** F1-score, Precision, Recall.  
- **Testing:** Evaluated on unseen infant cry audio samples.  

## 🛠️ Installation & Setup  

### Prerequisites  
Ensure you have the following installed:  
- Node.js  
- Python 3.x  

### Clone the Repository  
```bash
git clone https://github.com/yourusername/dont-cry.git
cd dont-cry
