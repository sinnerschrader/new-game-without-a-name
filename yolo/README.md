## YOLO_tensorflow

Tensorflow implementation of [YOLO](https://arxiv.org/pdf/1506.02640.pdf), including training and test phase.
Based on https://github.com/hizhangp/yolo_tensorflow.

### Installation

1. Inside the `yolo` directory execute `PIPENV_VENV_IN_PROJECT=true pipenv install` and `pipenv shell` to run the next commands in.

2. Download Pascal VOC dataset, and create correct directories
	```Shell
	$ bash ./download_data.sh
	```

3. Download [YOLO_small](https://drive.google.com/file/d/0B5aC8pI-akZUNVFZMmhmcVRpbTA/view?usp=sharing)
weight file and put it in `data/weight`

4. Modify configuration in `yolo/config.py`

5. Training
	```Shell
	$ python train.py
	```

6. Test
	```Shell
	$ python test.py
	```
