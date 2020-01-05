# CycleGAN-tf2
This project implements CycleGAN with tensorflow 2.0

## create dataset

create two directories "A" and "B" respectively under current directory. put dataset with style A in directory A and ones with style B in directory B. create dataset with the following command

```python
python3 create_dataset.py
```

after executing successfully, you can find two newly created tfrecord file under directory dataset.

## train

train the model with the following command

```python
python3 train.py
```
