from rouge import FilesRouge
import argparse

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('hyp', type=str, default="output/0_hyp.txt", help="Path to dataset")
    # parser.add_argument('ref', type=str, default="output/0_ref.txt", help="Path to val dataset")
    # args = parser.parse_args()
    # hyp = args.hyp
    # ref = args.ref
    for i in range(1):
        try:
            hyp = f"../eval/mimic_hyp.txt"
            ref = f"../eval/mimic_ref.txt"
            files_rouge = FilesRouge()
            scores = files_rouge.get_scores(hyp, ref, avg=True)
            rouge_1 = scores["rouge-1"]["f"]
            rouge_2 = scores["rouge-2"]["f"]
            rouge_L = scores["rouge-l"]["f"]
            print(f"rouge-1 {rouge_1}, rouge-2 {rouge_2}, rouge-l {rouge_L}")
        except ValueError:
            continue 