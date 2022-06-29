# Perceiver-Music-Generation

Create conditional pop music with perceiver-ar model implemented by [lucidrains](https://github.com/lucidrains/perceiver-ar-pytorch). 

## 1 Requirements
torch == 1.11.0 
transformers == 4.19.4 
pyarrow == 8.0.0


## 2 Dataset

Download the Magenta MAESTRO v.2.0.0 Piano MIDI Dataset from the [web](https://magenta.tensorflow.org/datasets/maestro), and put the file under the dir: 
```
./data
```
The music dataset is pre-processed with [midi neural pprocessor](https://github.com/jason9693/midi-neural-processor/tree/bea0dc612b7f687f964d0f6d54d1dbf117ae1307), do not worry, this code is integrated, and only needs to run:  
```base
$ python preprocess.py 
```

## 3 Train 
```base
$ python train.py 
```

## 4 Test 
We provide the trained ckpt in the google drive. 
```base
$ python generate.py 
```

