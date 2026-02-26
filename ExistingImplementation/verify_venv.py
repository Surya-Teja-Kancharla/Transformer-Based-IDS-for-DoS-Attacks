import numpy; import pandas; import cv2; import torch; import yaml
import sklearn, scipy, imblearn, matplotlib, tqdm, tensorboard

print('numpy   :', numpy.__version__)
print('pandas  :', pandas.__version__)
print('opencv  :', cv2.__version__)
print('torch   :', torch.__version__)
print('CUDA    :', torch.cuda.is_available())
print('GPU     :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print()
print('ALL OK — ready to train')