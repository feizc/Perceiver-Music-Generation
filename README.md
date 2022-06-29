# Perceiver-Music-Generation

Create conditional pop music with Perceiver-AR model implemented by [lucidrains](https://github.com/lucidrains/perceiver-ar-pytorch). 

## 1 Requirements

torch == 1.11.0 

transformers == 4.19.4 

pyarrow == 8.0.0


## 2 Dataset

Download the Magenta MAESTRO v.2.0.0 Piano MIDI Dataset from the [web](https://magenta.tensorflow.org/datasets/maestro), and put the file under the file direction: 
```
./data
```
The music dataset is pre-processed with [midi neural pprocessor](https://github.com/jason9693/midi-neural-processor/tree/bea0dc612b7f687f964d0f6d54d1dbf117ae1307), do not worry, the processing code is integrated in this repository, and only needs to run:  
```base
$ python preprocess.py 
```

## 3 Train 

```base
$ python train.py --data_dir [data path] --ckpt_dir [ckpt path]
```

## 4 Inference 
We provide the trained ckpt in the google drive. Download the trained ckpt and put it to the ckpt_dir, then run the command: 
```base
$ python generate.py --data_dir [data path] --ckpt_dir [ckpt path] --output_dir [output path]
```

Generated music cases can be found in [google drive](https://drive.google.com/file/d/1Ws4iIHGoD3TehZY2xxNF4kZ7pNt0vOAm/view?usp=sharing). 


