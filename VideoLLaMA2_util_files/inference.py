import os
import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer

from videollama2.utils import disable_torch_init

def inference(modal, modal_path, instruct, model, tokenizer, processor):
    return mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

if __name__ == "__main__":
    acc = {}
    near_far_map = {'near': 'near', 'far': 'far'}
    fh_map = {'fh': 'forehand', 'bh': 'backhand'}
    shot_map = {'IO': 'inside out', 'II': 'inside in', 'CC': 'cross-court', 'DL': 'down the line', 'DM': 'down the middle'}
    serve_map = {'B': 'body', 'W': 'wide', 'T': ' T '}
    stroke_map = {'serve': 'serve', 'stroke': 'stroke', 'return': 'return'}
    in_last_map = {'in': 'in', 'last': 'last', 'no in last' : 'no in last'}
    true_fh_bh = []
    true_shot = []
    true_serve = []
    true_stroke = []
    true_in_last = []
    true_near_far = []
    pred_fh_bh = []
    pred_shot = []
    pred_serve = []
    pred_stroke = []
    pred_in_last = []
    pred_near_far = []

    
    disable_torch_init()
    modal ='video'
    model_path = 'work_dirs_finetuned_clip_4/videollama2/finetune_downstream_sft_settings_qlora/checkpoint-7000/'
    model, processor, tokenizer = model_init(model_path)
    instruct = "What is happening in the tennis video?"
    test_dir = "./test_datasets"
    for dirs in os.listdir(test_dir):
        for files in os.listdir(os.path.join(test_dir, dirs)):
            print(files)
            try:
                stroke, shot, serve, fh_bh, near_far, in_last = ['no stroke', 'no shot', 'no serve', 'none_fh_bh', 'none', 'no in last']
                inf = inference(modal, os.path.join(test_dir, dirs, files), instruct, model, tokenizer, processor)
                print(inf)
                # get right mappings
                for i in fh_map:
                    if i in dirs:
                        fh_bh = fh_map[i]
                        break
                for i in shot_map:
                    if i in dirs:
                        shot = shot_map[i]
                        break
                for i in near_far_map:
                    if i in dirs:
                        near_far = near_far_map[i]
                        break
                for i in stroke_map:
                    if i in dirs:
                        stroke = stroke_map[i]
                        break
                for i in serve_map:
                    if i in dirs:
                        serve = serve_map[i]
                        break
                for i in in_last_map:
                    if i in dirs:
                        in_last = in_last_map[i]
                        break
                true_fh_bh.append(fh_bh)
                true_shot.append(shot)
                true_stroke.append(stroke)
                true_near_far.append(near_far)
                true_in_last.append(in_last)
                true_serve.append(serve)



                # get prediction mappings
                pred_stroke_temp, pred_shot_temp, pred_serve_temp, pred_fh_bh_temp, pred_near_far_temp, pred_in_last_temp = ['no stroke', 'no shot', 'no serve', 'none_fh_bh', 'none', 'no in last']
                for i in fh_map.values():
                    if i in inf:
                        pred_fh_bh_temp = i
                pred_fh_bh.append(pred_fh_bh_temp)
                for i in stroke_map.values():
                    if i in inf:
                        pred_stroke_temp = i
                pred_stroke.append(pred_stroke_temp)

                for i in shot_map.values():
                    if i in inf:
                        pred_shot_temp = i
                pred_shot.append(pred_shot_temp)

                for i in serve_map.values():
                    if i in inf:
                        pred_serve_temp  = i
                pred_serve.append(pred_serve_temp)
                  
                for i in near_far_map.values():
                    if i in inf:
                        pred_near_far_temp = i
                pred_near_far.append(pred_near_far_temp)
                  
                for i in in_last_map.values():
                    if i in inf:
                        pred_in_last_temp = i
                pred_in_last.append(pred_in_last_temp)
            except:
                continue

    print("true_fh_bh = ", true_fh_bh)
    print("true_shot = ", true_shot)
    print("true_stroke = ", true_stroke)
    print("true_in_last = ", true_in_last)
    print("true_near_far = ", true_near_far)
    print("true_serve = ", true_serve)
    print("pred_fh_bh = ", pred_fh_bh)
    print("pred_shot = ", pred_shot)
    print("pred_stroke = ", pred_stroke)
    print("pred_serve = ", pred_serve)
    print("pred_near_far = ", pred_near_far)
    print("pred_in_last = ", pred_in_last)

   

    
