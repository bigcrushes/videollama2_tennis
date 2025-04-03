import os
import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer

from videollama2.utils import disable_torch_init

def inference(modal, modal_path, instruct, model, tokenizer, processor):
    return mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

if __name__ == "__main__":
    disable_torch_init()

    prompts = {}
    with open("track_test_3.txt", 'r') as f:
        for line in f.readlines():
            name, word = line.split(' : ')
            prompts[name] = word

    modal ='video'
    model_path = 'work_dirs_keypoints_armslegs/videollama2/finetune_downstream_sft_settings_qlora/checkpoint-20000/'
    model, processor, tokenizer = model_init(model_path)
    test_dir = "./test_videos/"
    test_nums = {}
    for files in os.listdir(test_dir):
        try: 
            name = ''.join(files.split('.')[:-1])
            instruct = prompts[name]
            inf = inference(modal, os.path.join(test_dir, files), instruct, model, tokenizer, processor)
            print(files + ' : ' + inf, flush=True)
        except:
            continue
        
               
   

    
