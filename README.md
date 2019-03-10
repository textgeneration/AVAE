#Arrtributed controlled VAE Model
Implementing of Enhanced Attribute Controlled VAE Model in PyTorch.

## Requirements
1. Python 3.5+
2. PyTorch 0.3+
3. TorchText <https://github.com/pytorch/text>

## Dateset
Amazon product review dataset is available at <https://drive.google.com/open?id=13Hf7LSLCcua9sgWQfBQohv4b4VdlHezr>.

## How to run
1. Run `python train_vae.py --save {--gpu}`. This will create `vae.bin`. 
2. Run `python train_AttrVae --save {--gpu}`. This will create `AVae.bin`. 
3. Run `test.py --model {vae, AVae}.bin {--gpu}` for basic evaluations, e.g. conditional generation.
