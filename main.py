# Eğer freeze_support required hatası alınırsa
# multithread kütüphanesini dahil ediyoruz.

#import multiprocessing
from ultralytics import YOLOv10 # type: ignore

if __name__ == '__main__':
    # multiprocessing.freeze_support()
    
    model = YOLOv10('yolov10n.pt')
    model.train(data="dataset.yaml", epochs=20, batch=16, imgsz=640)
    model.val(data='dataset.yaml', batch=16)
