# Encoding Node Diffusion Competence and Role Significance for Network Dismantling


## Dependencies

- torch 1.7.1
- igraph 0.9.9
- networkx 2.6.3
- scipy 1.5.4
- sklearn 0.24.2
- numpy 1.19.5
- dgl 0.7.0

Install all dependencies using
```
pip install -r requirement.txt
```


## Usage
1.  Unzip the ```data.zip ``` and put them in the folder ```.\data``` . 
2.  Train and evaluate the model on real-world networks :

```
python train_realworld.py
```

3.  Train and evaluate the model on synthetic networks :

```
python train_synthetic.py --Type BA --N 1000 --M 4
```
`--Type` argument should be one of [ BA, WS, PLC, ER], `--N` set the size of synthetic networks, `--M` set the synthetic generation parameter.  

