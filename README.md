# Image Recognition

module CNN.py implements convolution neural network for image recognition learning.\
module evaluator.py can be used to see the success of pretrained neural network in CNN.py

trn.pkl - training dataset of images\
val.pkl - validation dataset of images

weights.pth - pretrained weights for the CNN

Top-n accuracy of current model on data from the same distribution as in trn.pkl and val.pkl with pretrained weights in weights.pth
| n | Accuracy | Error    |
|---|----------|----------|
| 1	| 59.800 % | 40.200 % |
| 2	| 75.200 % | 24.800 % |
| 3	| 83.600 % | 16.400 % |
| 4	| 89.200 % | 10.800 % |
| 5	| 92.800 % | 7.200 %  |
| 6	| 95.200 % | 4.800 %  |
| 7	| 97.000 % | 3.000 %  |
| 8	| 98.800 % | 1.200 %  |
| 9	| 99.600 % | 0.400 %  |
| 10| 100.000 %| 0.000 %  |
