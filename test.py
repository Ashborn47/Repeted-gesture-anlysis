"""test TransRAC model"""
import os
## if your data is .mp4 form, please use RepCountA_raw_Loader.py
# from dataset.RepCountA_raw_Loader import MyData
## if your data is .npz form, please use RepCountA_Loader.py. It can speed up the training
try:
    from dataset.RepCountA_Loader import MyData
    from models.TransRAC import TransferModel
    from testing.test_looping import test_loop
except:
    from modules import Modules



N_GPU = 1

device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
count,position = 0,None


video_path  = r"C:\Users\Asus\Downloads\Y2meta.app-41 pull ups-(480p).mp4"
root_path = r'/public/home/huhzh/LSP_dataset/LLSP_npz(64)/'

test_video_dir = 'test'
test_label_dir = 'test.csv'

# video swin transformer pretrained model and config
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
checkpoint = './pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'

# TransRAC trained model checkpoint, we will upload soon.
lastckpt = None
NUM_FRAME = 64
SCALES = [1, 4, 8]


output=Modules(video_path = r"C:\Users\Asus\Downloads\Y2meta.app - Push Up Challenge_ Can You Do 20 Push Ups in 15 Seconds.mp4",NUM_FRAME = 64, SCALES= [1, 4, 8])
print(output)

# multi scales(list). we currently support 1,4,8 scale.

