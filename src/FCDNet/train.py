import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/path to your model.yaml file')
    #model.load('/home/wld/ultralytics/yolov8n.pt') # loading pretrain weights
    model.train(data='/path to your dataset.yaml file',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=32,
                #close_mosaic=10,
                #workers=8,
                #device='0',
                #optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train',
                #name='',
                )		
