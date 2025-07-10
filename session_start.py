import fiftyone as fo
import argparse

datasets = fo.list_datasets()
dataset = fo.load_dataset(datasets[0])

parser = argparse.ArgumentParser(description="Port settings")
parser.add_argument("--port", required= True, type = int, help = "Port to start a fiftyone app session.")
args = parser.parse_args()
port_num = args.port

session = fo.launch_app(dataset, remote=True, address="0.0.0.0", port = port_num)
session.wait()
