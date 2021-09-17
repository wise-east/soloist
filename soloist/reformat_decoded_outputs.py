import json 
from argparse import ArgumentParser

parser= ArgumentParser()

parser.add_argument("-fn", "--filename", help="file to reformat such that we only keep the belief states") 

args = parser.parse_args()

with open(args.filename, "r") as f: 
    data = json.load(f)

# just need to drop the first item 
data = [item[1:] for item in data] 

with open(args.filename.replace(".json", "") + "_reformatted.json", "w") as f: 
    json.dump(data, fp=f, indent=2)

    
