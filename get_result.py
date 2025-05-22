import os 
import sys 
import re
base_path = sys.argv[1]
print(base_path)
eval_file_names = os.listdir(base_path)
eval_file_names = [i for i in eval_file_names if "_eval_log" in i]
eval_file_names.sort(key=lambda x: int(re.search(r'checkpoint_(\d+)_eval_log', x).group(1)))
with open(os.path.join(base_path,"total_eval_result"),"w") as f1:
    for eval_file_name in eval_file_names:
        f1.write(eval_file_name+"\n")
        eval_file_path = os.path.join(base_path,eval_file_name)
        with open(eval_file_path) as f:
            lines = f.readlines()
            for line in lines:
                if "ACC" in line:
                    line = line.strip().split(" ACC: ")
                    name,score = line[0],line[1]
                    f1.write(name+" ACC:\t"+score+"\n")

                



