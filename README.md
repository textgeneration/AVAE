## Requirements
1. Python 3.5+
2. PyTorch 0.3
3. TorchText <https://github.com/pytorch/text>

## How to run
1. SETTINGS:ALL configuration of this project is in CONFIG.py, including file path, latent variable's dimension or batch size.
When you have configured this parameters, you can train the model.
2. Run `python train_vae.py `. This will create `vae_c-dimension_iteraion`. Essentially this is the base VAE.
3. Run `python train_avae.py`. This will create `avae_c-dimension_iteration`. 
4. Run `test.py ` for basic evaluations, for example random generate sentence.

## Tips
1. We use review dataset consists of some meaningless symbol and you need to clean the dataset. We offer a tools avae/preprocess.py and also offer the text data that has been clean, which can be download from <https://drive.google.com/drive/folders/1DQBaaoV6cCZD2uLe-zLay0YfmdjHR6sq?usp=sharing>. If you need the original review dataset, you can download them from the official website.
2. We provide two pretrained well generative model to generation sentiment and product categories with ratings respectively. You can run different model using test.py, and the model can download data automaticlly. The model can also be download from <https://drive.google.com/drive/folders/1DQBaaoV6cCZD2uLe-zLay0YfmdjHR6sq?usp=sharing>, C_DIMENSION in CONFIG.py can control which model you want to run.