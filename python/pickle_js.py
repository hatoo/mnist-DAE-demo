import pickle
import sys
import json

if __name__ == '__main__':
    input_file  = sys.argv[1]
    with open(input_file) as f:
        data = pickle.load(f)
        for k in data.keys():
            print "{0} = {1};\n".format(k, json.dumps(data[k].tolist()))