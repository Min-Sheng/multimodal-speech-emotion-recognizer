# multimodal-speech-emotion-recognizer


## This is a PyTorch re-implementation code for the following paper:
**Multimodal Speech Emotion Recognition using Audio and Text**, IEEE SLT-18, <a href="https://arxiv.org/abs/1810.04635">[paper]</a>

----------

### [download data corpus]
- IEMOCAP <a href="https://sail.usc.edu/iemocap/">[link]</a>
<a href="https://link.springer.com/article/10.1007/s10579-008-9076-6">[paper]</a>
- download IEMOCAP data from its original web-page (license agreement is required)

### [data preprocessing]
- for the preprocessing, refer to codes in the "./preprocessing"
- this part comes from the paper auther's repository: https://github.com/david-yoon/multimodal-speech-emotion
- Examples
	> MFCC : MFCC features of the audio signal (ex. train_audio_mfcc.npy) <br>
	> MFCC-SEQN : valid lenght of the sequence of the audio signal (ex. train_seqN.npy)<br>
	> PROSODY : prosody features of the audio signal (ex. train_audio_prosody.npy) <br>
	> LABEL : targe label of the audio signal (ex. train_label.npy) <br> 
	> TRANS : sequences of trasnciption (indexed) of a data (ex. train_nlp_trans.npy) <br>

### [training]
- run "train_script.sh"
- or run "python "single_text_trainer.py", "single_audio_trainer.py", "multi_modal_trainer.py", and "multi_modal_attn_trainer.py", manually

## [evaluation]
- run "evaluate.ipynb" to inference testing data with 4 different models
- and then, run "analysis/Confusion_Matrix.ipynb" to plot the confusion matrix
