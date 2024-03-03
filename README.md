# Instructions to run code:
The fine-tuned VGG16 and VGG19 models on the breast histopathology dataset, as well as a subset of the preprocessed dataset can be found at - 
https://drive.google.com/drive/folders/1AnsWEyuoFxGag3uOurpP2m_rs3Ly2Kbc?usp=sharing

- To generate the 3D t-SNE plots, please change the paths to the model checkpoint and dataset in the file tsne.py (marked by # TODO) and run the file.

- To generate saliency maps for a 20 random samples from the test set, please change the paths to the model checkpoint and dataset in the file saliency.py (marked by # TODO) and run the file.

- To train a model, please change the path to the data in train.py (marked by #TODO) and change hyperparameters as desired and run the file. 
Note: the checkpoints mentioned above were trained on a larger subset of the breat histopathology data, however only a portion of that dataset was uploaded to drive due to memory limitations.
