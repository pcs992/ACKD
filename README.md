# Asymmetric Cross-Modal Knowledge Distillation: Bridging Modalities with Weak Semantic Consistency

## 📁 Project Structure

├── configuration/ # Configuration files 

├── datasets/ # Teacher-student paired list

├── Distillers/ # Knowledge distillation approaches 

├── log/ # Training logs and outputs 

├── model/ # Model architecture definitions 

├── utils/ # Utility functions and helpers 

├── weights/ # Model checkpoints 

├── train_matcher.py # Step 1: Train the matcher model 

├── generator.py # Step 2: Generate matched datasets

├── train_teacher.py # Step 3: Train teacher model 

├── train_student.py # Step 4: Train student model (our agent, SemBridge) 

├── val.py # Validation script 

└── requirements.txt # Python dependencies

## 1. Download dataset benchmark
https://drive.google.com/drive/u/1/folders/1IuEDQv7yNsqxX5fEaX-kyKdXlYfH5fC7
## 2. Install dependencies
pip install -r requirements.txt

## 3. Training matcher model
python train_matcher.py --dataset S2S_EU

## 4. Generating matched datasets initially
python generator.py --dataset S2S_EU

## 5. Training the teacher model
python train_teacher.py --dataset S2S_EU

## 6. Training the student model with our agent, SemBridge
python train_student.py --dataset S2S_EU

## 📌 Notes
Make sure your Python version is 3.7+.

Update [Your Path] in main_config.py and configration/config_matcher.py for each datasets.

Logs and model weights are saved in log/ and weights/.
