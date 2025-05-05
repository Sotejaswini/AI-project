# AI-project
# 🧠 AI Sudoku Solver Using OCR 🎯  

## 📖 Overview  
This project is an **AI-powered Sudoku Solver** that leverages **Optical Character Recognition (OCR)** and **Deep Learning** to automatically detect, extract, and solve Sudoku puzzles from images. It uses **computer vision** to preprocess images, a **CNN-based model** to recognize digits, and a **backtracking algorithm** to solve the Sudoku grid.  

---

## 🚀 Features  
✅ **Digit Recognition:** Uses a trained deep learning model to recognize numbers in the Sudoku grid.  
✅ **Image Processing:** Uses OpenCV to extract and preprocess Sudoku grids from images.  
✅ **Sudoku Solving Algorithms:** Implements optimized backtracking and constraint propagation algorithms.  
✅ **Pre-trained Models:** Includes pre-trained models for easy deployment.  
✅ **Supports Custom Input Images:** Can solve Sudoku puzzles from scanned or captured images.  

---

## 🛠️ Tech Stack  
- **Python 3.x**  
- **Keras** (for deep learning model)  
- **OpenCV** (for image processing)  
- **NumPy** (for matrix operations)  
- **Matplotlib** (for visualization)  
- **Scikit-Learn** (for machine learning utilities)  
- **Imutils** (for image transformations)  

---
#Projec Structure
📂 AI-Sudoku-Solver  
├── 📂 model/               # Model training and pre-trained models  
│   ├── train_model.py      # Script to train the digit recognition model  
│   ├── model.h5            # Trained Keras model for digit recognition  
│   ├── pretrained_models/  # Folder with pre-used models  
│   └── ...  
│  
├── 📂 algorithm/            # Contains different Sudoku-solving algorithms  
│   ├── backtracking.py      # Backtracking-based solver  
│   ├── constraint_solver.py # Constraint propagation solver  
│   ├── ...  
│  
├── 📂 stats_calculation/     # Additional files for testing model performance  
│  
├── main.py                 # Main script to process and solve Sudoku  
├── requirements.txt        # List of dependencies  
├── README.md               # Project documentation  
└── howtouse.txt            # Additional testing instructions  
```

---

## 🔧 Installation & Setup  

### **Step 1: Clone the Repository**  
```bash
git clone https://github.com/Sotejaswini/AI-project.git
cd AI-project
```

### **Step 2: Install Dependencies**  

```bash
pip install keras
pip install opencv-python  # This is what "computer vision" refers to
pip install imutils
pip install matplotlib
pip install numpy
pip install scikit-learn   # sklearn is installed as scikit-learn
```

### **Step 3: Train the Model (If Needed)**  
To train the digit recognition model, run:  
```bash
python model/train_model.py
```
- This will create a trained model (`model.h5`) in the `model/` folder.  
- Pre-trained models are already provided if you want to skip training.  

---

## 🎮 How to Run  

### **Solve a Sudoku Puzzle from an Image**  
Run the following command:  
```bash
python main.py <path_to_input_image> <path_to_model>
```

#### **Example:**  
```bash
python main.py ./input/sudoku.jpg ./model/model.h5
```
- The solved Sudoku will be saved as **`solved_sudoku.jpg`**.  

---

## 📊 Testing & Evaluation  
The `stats_calculation` folder contains additional scripts for performance testing.  
Refer to **`howtouse.txt`** for details on how to use these scripts.  

---

## 🏆 Results  
- The trained model achieves **high accuracy** in recognizing Sudoku digits.  
- The Sudoku solver can process **real-world images** and provide correct solutions.  
- **Performance Metrics** are available in the **`stats_calculation`** folder.  

---

### 🖥️ Terminal Output Example
![Model Output](Main%20Model%20and%20supporting%20files/screenshots/output_2.png)
![Screenshot](https://github.com/Sotejaswini/AI-project/blob/main/AI-SUDOKU-SOLVER-USING-OCR-main/Main%20Model%20and%20supporting%20files/screenshots/output_2.png?raw=true)

