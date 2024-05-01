from argparse import ArgumentParser
import os
import random

from tqdm import tqdm
import pandas as pd
import cv2

IMG_PATH = "/mnt/c/Users/Public/LARD/LARD_test_synth/images"

class DatasetGenerator():
    
    def __init__(self, input_dir: str, output_dir: str, test_ratio: float, train_ratio: float):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.test_ratio = test_ratio
        self.train_ratio = train_ratio
        self.output_test_rwy_dir = os.path.join(output_dir, "test/rwy") 
        self.output_test_norwy_dir = os.path.join(output_dir, "test/norwy") 
        self.output_train_rwy_dir = os.path.join(output_dir, "train/rwy") 
        self.output_train_norwy_dir = os.path.join(output_dir, "train/norwy") 
        self.output_val_rwy_dir = os.path.join(output_dir, "val/rwy") 
        self.output_val_norwy_dir = os.path.join(output_dir, "val/norwy") 
        
    def run(self):
        self.__makedirs()
        self.__read_index()
        self.__split()
        self.__generate("test")
        self.__generate("train")
        self.__generate("val")
              
    def __makedirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_test_rwy_dir, exist_ok=True)
        os.makedirs(self.output_test_norwy_dir, exist_ok=True)
        os.makedirs(self.output_train_rwy_dir, exist_ok=True)
        os.makedirs(self.output_train_norwy_dir, exist_ok=True)
        os.makedirs(self.output_val_rwy_dir, exist_ok=True)
        os.makedirs(self.output_val_norwy_dir, exist_ok=True)
        
    def __read_index(self):
        print("Load LARD dataset index file")
        dirname = os.path.basename(self.input_dir)
        csv_name = os.path.join(self.input_dir, f"{dirname}.csv")
        self.df = pd.read_csv(csv_name, sep=";")
    
    def __split(self):
        print("Split based on airport")
        airports = self.df.airport.unique()
        nb_test = round(self.test_ratio * len(airports))
        random.seed(1975)
        if self.test_ratio > 0:
            test_airports = random.choices(airports, k=nb_test)
            airports = list(set(airports) - set(test_airports))
            print(f"test airports: {test_airports}")
        nb_val = round((1 - self.train_ratio) * len(airports))
        val_airports = random.choices(airports, k=nb_val)
        print(f"val airports: {val_airports}")
        print(f"train airports: {list(set(airports) - set(val_airports))}")
        
        def add_dataset(row):
            if row['airport'] in test_airports:
                return "test"
            elif row['airport'] in val_airports:
                return "val"
            else:
                return "train"
        self.df["dataset"] = self.df.apply(add_dataset, axis=1)
        self.df.to_csv(os.path.join(self.output_dir, f"{os.path.basename(self.input_dir)}.csv"))

    def __generate(self, dataset: str):
        print(f"Generate images for {dataset} dataset")
        df = self.df.query(f"dataset == '{dataset}'")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_path = os.path.join(self.input_dir, row["image"])
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            # get bounding box of the label
            min_x = min(row["x_A"], row["x_B"], row["x_C"], row["x_D"])
            max_x = max(row["x_A"], row["x_B"], row["x_C"], row["x_D"])
            min_y = min(row["y_A"], row["y_B"], row["y_C"], row["y_D"])
            max_y = max(row["y_A"], row["y_B"], row["y_C"], row["y_D"])
            # crop the area corresponding to the label
            crop_rwy_img = img[min_y:max_y, min_x:max_x]
            cv2.imwrite(os.path.join(self.output_dir, f"{dataset}/rwy/{img_name}"), crop_rwy_img)
            # generate other images for the non runway class
            img_h, img_w, _ = img.shape
            label_h = max_y - min_y
            label_w = max_x - min_x
            margin = min(label_h, label_w)
            # generate image on the left of the label, if there is enough space
            left_x = min_x - label_w - margin
            if left_x > 0:
                crop_left_img = img[min_y:max_y, left_x:min_x-margin]
                cv2.imwrite(os.path.join(self.output_dir, f"{dataset}/norwy/LEFT-{img_name}"), crop_left_img)
            # generate image on the right of the label, if there is enough space
            right_x = max_x + label_w + min(label_h, label_w)
            if right_x < img_w:
                crop_right_img = img[min_y:max_y, max_x+margin:right_x]
                cv2.imwrite(os.path.join(self.output_dir, f"{dataset}/norwy/RIGHT-{img_name}"), crop_right_img)

if __name__ == "__main__":
    parser = ArgumentParser(description='Create a dataset')
    parser.add_argument('-i','--input', help='Input directory', required=True)
    parser.add_argument('-o','--output', help='Output directory', default="/mnt/c/Users/Public/RWC")
    parser.add_argument('--test', help='Test ratio', default=0.3)
    parser.add_argument('--train', help='Train ratio (after removing test data)', default=0.8)
    args = parser.parse_args()
    
    print(args.input)
    DatasetGenerator(args.input, args.output, args.test, args.train).run()