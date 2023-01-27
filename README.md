# Encoding Node Diffusion Competence and Role Significance for Network Dismantling

The official PyTorch implementation of DCRS in the following paper:

```
Jiazheng Zhang, Bang Wang. 2023. Encoding Node Diffusion Competence and Role Significance for Network Dismantling.
In WWW'23, April 30 – May 4, 2023, Austin, Texas, USA, 12 pages.
```

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

## Citation

Please cite our work if you find our code/paper is helpful to your work.

```
@inproceedings{zhang2023DCRS,
  title={Encoding Node Diffusion Competence and Role Significance for Network Dismantling},
  author={Zhang, Jiazheng and Wang, Bang},
  booktitle={Proceedings of the 2023 ACM Web Conference},
  series={WWW'23},
  year={2023},
  location={Austin, Texas, USA},
  numpages={12}
}
```
