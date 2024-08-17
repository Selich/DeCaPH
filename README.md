# DeCaPH: Decentralized Collaborative and Privacy-Preserving Machine Learning for Multi-Hospital Data

## Overview

This project demonstrates the implementation of a decentralized, collaborative, and privacy-preserving machine learning framework tailored for multi-hospital data. The goal is to enable multiple hospitals to collaboratively train a machine learning model on their private datasets without sharing the raw data, thereby preserving patient privacy.

Key components of this project include:

- **Decentralized Training**: No central server is required. A leader is randomly selected in each round to aggregate model updates.
- **Differential Privacy**: The project employs differential privacy techniques, including gradient clipping and noise addition, to ensure that the privacy of individual data points is maintained.
- **Secure Aggregation**: Gradients from individual hospitals are securely aggregated, ensuring that no single entity can access raw gradients from other participants.
- **Heterogeneous Data Distribution**: The project simulates slightly different data distributions across hospitals to reflect real-world variations in case prevalence.

## Citation

This project is based on the framework proposed in the paper:

**Fang, C., Dziedzic, A., Zhang, L., Oliva, L., Verma, A., Razak, F., Papernot, N., & Wang, B.** *Decentralized, Collaborative, and Privacy-preserving Machine Learning for Multi-Hospital Data*. The framework enables multiple hospitals to collaborate on training machine learning models without directly sharing their private data and while preserving patient privacy through differential privacy techniques.


## Dataset

This project uses the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle. The dataset includes X-ray images labeled as COVID-19, Normal, and Viral Pneumonia.

The dataset is split across multiple hospitals to simulate a scenario where each hospital has slightly different data distributions.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-repository/DeCaPH_Project.git
cd DeCaPH_Project
```

### 2. Install Dependencies
Ensure you have Python 3.8 or above installed. Then, install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
Download the COVID-19 Radiography Database from Kaggle.
Organize the dataset into separate folders for each hospital inside the data/ directory.
Run the data splitting script to distribute the data among hospitals:

```bash
python -c "from utils.data_splitter import load_all_data, split_data_heterogeneous, save_hospital_data; data = load_all_data('path_to_your_COVID-19_Radiography_Database'); hospital_splits = split_data_heterogeneous(data, num_hospitals=3); save_hospital_data(hospital_splits)"
```

### 4. Run the Training
Once the data is prepared, run the training script:

```bash
python main.py
```

This will start the decentralized training process, where each hospital contributes to the training of a CNN model while preserving the privacy of its data.

### Contributing
If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are warmly welcome.

### Acknowledgments
- The COVID-19 Radiography Database is provided by a collaboration of medical professionals and AI researchers and is hosted on Kaggle.
- This project is inspired by the DeCaPH framework for privacy-preserving federated learning in healthcare, as described in the paper cited above.