import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/path to your model.pt') # select your model.pt path
    model.predict(source='',
                  imgsz=640,
                  project='runs/detect',
                  name='',
                  save=True,
                  stream_buffer = True,
                  #visualize=True # visualize model features maps
                )
