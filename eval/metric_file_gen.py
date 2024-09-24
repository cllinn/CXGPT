import json
import os

def gen(res, gts):
    res = open(res, "r")
    gts = open(gts, "r")
    res_dict = {}
    gts_dict = {}
    for i, line in enumerate(res):
        res_dict[i] = [line]
    for i, line in enumerate(gts):
        gts_dict[i] = [line]
    
    with open("/root/mulcon/eval/metrics/CaptionMetrics/examples/gts.json", "w") as f:
        f.write(json.dumps(gts_dict, sort_keys=True, indent=4, separators=(',', ': ')))
    with open("/root/mulcon/eval/metrics/CaptionMetrics/examples/res.json", "w") as f:
        f.write(json.dumps(res_dict, sort_keys=True, indent=4, separators=(',', ': ')))

        
if __name__ == "__main__":
    res = "rg_hyp.txt"
    gts = "rg_ref.txt"
    gen(res, gts)