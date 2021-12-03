#PSSGR(Person-Specific Search and Graph Representation)
This is the official code of the ACM MM 2021 oral paper：

Personality Recognition by Modelling Person-specific Cognitive Processes using Graph Representation.
Shao, Z., Song, S*., Jaiswal, S., Shen, L., Valstar, M. and Gunes, H., 2021. 

##Dependencies:
Please install the following libraries (Using conda is highly recommended)
<br>ffmpeg
<br>python_speech_features
<br>scipy
<br>pytorch-gpu
<br>torchvision
<br>tqdm
<br>h5py
<br>tensorboard

##Preparing your data
* 1. Please use OpenFace to process all pair of video data, where aligned faces are produced and the facial landmarks are saved in the CSV files.

The example of the obtained data:
：
```
|-root
|   |-talk1
|   |   |-Expert_video_aligned
|   |   |-Novice_video_aligned
|   |   |-Expert_video.csv
|   |   |-Novice_video.csv
|   |-talk2
|   |   |-Expert_video_aligned
|   |   |-Novice_video_aligned
|   |   |-Expert_video.csv
|   |   |-Novice_video.csv
|   |-...
|   |-...
|   |-...
|   |-talkn
|   |   |-Expert_video_aligned
|   |   |-Novice_video_aligned
|   |   |-Expert_video.csv
|   |   |-Novice_video.csv
```
* 2. Once you prepared your data, please use the following command to align video-audio signals and package them in pickle data structure：

     python process_noxi_data.py --vidio_dir ‘the folder contains soruce audio files’ --openface_dir ‘openface folder’ --save_dir ‘audio feature folder’


##Training Supernet
Please run the following command line:
python train_supernet.py --ID XX --gpu_id XXX --data XXX <BR>
--ID: 'the ID of the video ID'<br>
--gpu_id: ’GPU ID‘<br>
--data: ’The folder containing the prepared training data ‘<br><br>

