## Finetue MobileNetV2
1. Download training data, unzip to training_data forlder
2. Training data structure as following
    .
    ├── train                  # Training data
    ├── test                   # Test data
    ├── valid                  # Data for verify 
    ├── birds.csv              # Labal file for MobileNetV2
3. RUNPATH cardinalis project
4. active training environment 
   ````
    cd ./models
    pipenv shell
    cd ../
    python ./models/mobilenet_v2/train.py
    ```

   