Main file: 
    Amazon_Latest_Code.ipynb
        Note: Make sure utils.py is imported correctly
Other files: 
    utils.py
    
Dataset: https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset
    Note: Make sure dataset path is set correctly - I used Electronics data for my project
    
Running on google colab: (recommended)
    !pip install contractions
    !pip install pyLDAvis==2.1.2
    !pip install datasets
    !pip install transformers[torch]
    !pip install accelerate -U

Running locally via conda: (assuming conda is installed)
    conda new -n <env_name>
    conda activate <env_name>
    pip install -r requirements.txt

To train models on other architectures, make sure to replace the LSTM architecture with your own architecture.