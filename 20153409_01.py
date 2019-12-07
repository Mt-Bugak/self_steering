from tensorflow import keras
import h5py
import os, cv2
import argparse

parser = argparse.ArgumentParser(description='python 20153409_3.py --input_dir ./input_data_folder --output_dir ./output_data_folder')
parser.add_argument('--input_dir', type=str, help='put an input directory')
parser.add_argument('--output_dir', type=str, help='put an output directory')
args = parser.parse_args()

input_list = os.listdir(args.input_dir)

model = keras.models.load_model('20153409_01.h5')
for i in input_list:
  if(i[0]=='.'): continue
  img = cv2.imread(args.input_dir+'/'+i)
  predicted_steers = model.predict(img[None, :, :, :].transpose(0, 3, 1, 2))[0][0]
  print(round(predicted_steers,5))
