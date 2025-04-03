# Using VideoLLaMA2 for tennis video analysis
Following are what each folder and file is for:
*clip_finetuning*: Contains files for finetuning CLIP model separately
*stc_connector*: Contains files for finetuning VideoLLaMA2's STC connector separately
*videollama2*: Mostly same as VideoLLaMA2, with minor edits in some files to adjust trainable parameters and add finetuned CLIP seaprately. Changes moslty in train.py and model/encoder.py
*VideoLLaMA2_util_files*: Contains util files to be used for VideoLLaMA2, mainly different inference scripts
*data_setup*: Sets up json file for training
*download_video*: Helps download videos from online
*other_models*: Tools for combining VideoLLaMA2 with other models
*results*: Used for analysing our results (e.g. calculating edit scores, confusion matrices)
