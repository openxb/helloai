from tqdm import tqdm
import os
import csv
from argparse import ArgumentParser

from ultralytics import YOLO
import numpy as np

class ModelTester():
    
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.model = YOLO("last.pt")
        
    def run(self):
        with open("out.csv", "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            
            for image_dir in tqdm([ f.path for f in os.scandir(self.input_dir) if f.is_dir() ]):
                for image_file in tqdm(os.listdir(image_dir)):
                    writer.writerow([
                        image_file,
                        os.path.basename(image_dir),
                        self.run_image(os.path.join(image_dir, image_file))
                    ])
    
    def run_image(self, image: str) -> str:
        results = self.model(image, )  # predict on an image
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()
        return names_dict[np.argmax(probs)]

if __name__ == "__main__":
    parser = ArgumentParser(description='Create a dataset')
    parser.add_argument('-i','--input', help='Input directory', required=True)
    args = parser.parse_args()
    
    print(args.input)
    ModelTester(args.input).run()