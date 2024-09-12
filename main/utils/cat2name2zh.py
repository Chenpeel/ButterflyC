import os
import json
import subprocess

def cat2namePrint(json_path):
    print_name = ""
    with open(json_path, "r") as f:
        cat2name = json.load(f)
        for i in range(1, 103):
            print_name += cat2name[str(i)] + ","
    print(print_name)
    subprocess.run("pbcopy", text=True, input=print_name)
    return print_name

def cat2namezhRewrite(input_zh, output_zh, json_path):
    '''
    Rewrite json type:
    {
        '1': ['pink primrose','粉色报春花'],
        ...
    }
    input_zh only ONE line, format:
    zh-word,zh-word,...,zh-word
    
    '''
    if not os.path.exists(output_zh):
        os.system(f"touch {output_zh}")
    cat2nameWzh = {}
    zh_names = input_zh.split(",")
    with open(json_path, "r") as f:
        cat2name = json.load(f)
        for i in range(1, 103):
            cat2nameWzh[str(i)] = [cat2name[str(i)], zh_names[i-1]]
    with open(output_zh, "w") as f:
        json.dump(cat2nameWzh, f, ensure_ascii=False)
    print("Rewrite Success")

def __main__():
    os.chdir("/Users/alpha/Desktop/selfRepo/FlowerRecognition/")
    json_path = "./data/cat2name.json"
    print("Gen all flower names(Copied to clipboard)")
    cat2namePrint(json_path)
    continue_flag = input("y: Input the ZH words , n: Exit\n[y/n]:")
    if continue_flag == "y":
        print("Input Rewrite ZH names")
        # use google translate api to translate english names to chinese
        names_en = cat2namePrint(json_path)
        names_zh = input("Input ZH names\n")
    else :
        exit(0)

if __name__ == "__main__":
    __main__()


