# SegRoot
Source code for SegRoot, a [paper](https://www.sciencedirect.com/science/article/pii/S0168169919303874) titled "SegRoot: a high throughput segmentation method for root image analysis"
<img align="center" src="https://github.com/wtwtwt0330/SegRoot/raw/master/examples/Untitled_T051_L004_10.04.17_131720_003_Untitled.jpg" width="350" height="350">
<img align="center" src="https://github.com/wtwtwt0330/SegRoot/raw/master/examples/Untitled_T051_L004_10.04.17_131720_003_Untitled-pre-mask-segroot-64,5.jpg" width="350" height="350">
# Contents
* Source code for training SegRoot networks (/code)
* A training set of 65 soybean root images and corresponding annotations (/data/data_raw)
* A pretrained SegRoot-8,5 weights (/weights)
* A test.jpg image for prediction (/data/prediction)
# Requirements
* Ubuntu / macOS / Windows
* Python3
* PyTorch 0.4.0 and above
* Scikit-Learn and Scikit-Image
# Usage
* clone this repository
```
git clone https://github.com/wtwtwt0330/SegRoot.git
```

* go to the code folder and run preprocessing script
```
cd code
python binarize_crop.py
```

* predict a test image using pretrained weights of SegRoot-8,5
```
python predict_images.py
```
you can also put your own images in the /data/prediction/ folder and run something like this:
```
python predict_images.py --image <YourImageFilename>
```
or you can predict all the images in a folder using:
```
python predict_images.py --all --data_dir <YourImageFolderPath>
```

* training a new SegRoot, the default setting is to train a SegRoot-8,5
```
python main_segroot.py --width <4,8,16,32,64> --depth <4,5>
```
see more details by using ```python main_segroot.py -h```,
the trained weights will be stored in /weights folder

* after training a new SegRoot, you can load the trained weights to predict images:
```
python predict_images.py --weights ../weights/<WeightsFilename>
```
make sure the width and depth are the same as the trained SegRoot network.
