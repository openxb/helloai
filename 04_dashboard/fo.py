import argparse
import os
import fiftyone as fo
import pandas as pd

def create_dataset() -> fo.Dataset:
    dataset = fo.Dataset.from_dir(
        "/mnt/c/Users/Public/RWC/test",
        fo.types.ImageClassificationDirectoryTree,
        name="runway_classifier-TEST",
        overwrite=True,
    )
    
    predictions_file = "../03_inference/out.csv"
    if os.path.isfile(predictions_file):
        df = pd.read_csv(predictions_file)
        for sample in dataset:
            sample["test_result"] = df[df.image_path == sample["filepath"]]["test_result"].unique()[0]
            sample.save()
            
    dataset.compute_metadata()
    
    print(dataset.first())
    
    dataset.persistent = True
    return dataset

def load_dataset() -> fo.Dataset:
    return fo.load_dataset("runway-classifier")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", help="List the datasets",
                    action="store_true")
    parser.add_argument("--clean", help="List the datasets",
                    action="store_true")
    parser.add_argument("--create", help="Create the dataset",
                    action="store_true")
    parser.add_argument("--display", help="Display the dataset",
                    action="store_true")
    args = parser.parse_args()
    
    if args.list:
        print(fo.list_datasets())
    
    if args.clean:
        for name in fo.list_datasets():
            print(f"delete dataset: {name}")
            fo.delete_dataset(name)
    
    if args.create:
        dataset = create_dataset()
        session = fo.launch_app(dataset)
        session.wait()
    
    if args.display:
        dataset = load_dataset()
        session = fo.launch_app(dataset)
        session.wait()