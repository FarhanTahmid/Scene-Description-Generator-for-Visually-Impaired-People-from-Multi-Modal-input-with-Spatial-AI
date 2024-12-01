
from __future__ import print_function
import sys
import os
import pickle
import PIL
import skimage.io as io
import torch

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager

from scenedetect.detectors.content_detector import ContentDetector
from scenedetect.detectors.threshold_detector import ThresholdDetector

from typing import List, Tuple

import cv2

BASE_DIR="G:/OneDrive - northsouth.edu/CODES/PROJECTS/PROJECT - Scene Description Generator for Visually Impaired People from Multi Modal Input With Spatial AI/"

sys.path.append(BASE_DIR+'software_utils/')

# To account for path errors
try:
    from .model.image_captioner import ImageCaptioner
    from .model.video_captioner import VideoCaptioner
    from .model.encoderCNN import EncoderCNN
except ImportError:
    pass

try:
    from .software_utils.create_transformer import create_transformer
    from .software_utils.vocabulary import Vocabulary
except ImportError:
    pass

class BySceneGenerator(object):
    img_extensions = ['jpg', 'png', 'jpeg']
    vid_extensions = ['mp4', 'avi']

    def __init__(self,
                 root_path="",
                 msrvtt_vocab_path="generator_app/Data/processed/msrvtt_vocab.pkl",
                 base_model='resnet152',
                 vc_model_path="generator_app/model/video_model/video_caption-model11-110-0.3354-5.0.pkl",
                 vid_embedding_size=2048,
                 embed_size=256,
                 hidden_size=512,
                 num_frames=40,
                 max_caption_length=35,
                 vc_rnn_type='lstm',
                 im_res=224):

        # Store class variables
        self.num_frames = num_frames
        self.vid_embedding_size = vid_embedding_size
        self.max_caption_length = max_caption_length

        # Load vocabularies
        with open(root_path + msrvtt_vocab_path, 'rb') as f:
            self.msrvtt_vocab = pickle.load(f)

        # Load transformer and image encoder
        self.transformer = create_transformer()
        self.encoder = EncoderCNN(base_model)

        # Create video captioner and load weights
        self.video_captioner = VideoCaptioner(
            vid_embedding_size,
            embed_size,
            hidden_size,
            len(self.msrvtt_vocab),
            rnn_type=vc_rnn_type,
            start_id=self.msrvtt_vocab.word2idx[self.msrvtt_vocab.start_word],
            end_id=self.msrvtt_vocab.word2idx[self.msrvtt_vocab.end_word]
        )

        if torch.cuda.is_available():
          vc_checkpoint = torch.load(root_path + vc_model_path)
        else:
          vc_checkpoint = torch.load(root_path + vc_model_path, map_location='cpu')
        self.video_captioner.load_state_dict(vc_checkpoint['params'])

        # Push all torch models to GPU / set on eval mode
        if torch.cuda.is_available():
            self.encoder.cuda()
            self.video_captioner.cuda()

        self.encoder.eval()
        self.video_captioner.eval()
    
    
    def find_scene_changes(self,video_path: str, method: str = 'threshold', new_stat_file: bool = True) -> List[Tuple[int, int]]:
        """
        Detect scene changes in a given video.

        Args:
            video_path: Path to the video to analyze.
            method: Method for detecting scene changes ('content' or 'threshold').
            new_stat_file: Whether to create a new stats file.

        Returns:
            List of scene changes with their corresponding frame ranges as tuples.
        """
        # Initialize the VideoManager and StatsManager.
        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()

        # Create a SceneManager and add the appropriate detector.
        scene_manager = SceneManager(stats_manager)
        if method == 'content':
            scene_manager.add_detector(ContentDetector(threshold=30, min_scene_len=40))
        else:
            scene_manager.add_detector(ThresholdDetector(threshold=125, min_scene_len=40))

        # Set the path for the stats file.
        stats_file_path = f'{video_path}.{method}.stats.csv'
        scene_list = []

        try:
            # Load stats file if it exists and new_stat_file is False.
            if not new_stat_file and os.path.exists(stats_file_path):
                with open(stats_file_path, 'r') as stats_file:
                    stats_manager.load_from_csv(stats_file)

            # Set the downscale factor for faster processing.
            video_manager.set_downscale_factor(2)

            # Start the video manager.
            video_manager.start()

            # Perform scene detection.
            scene_manager.detect_scenes(video_manager)

            # Obtain the list of scenes.
            scene_list = scene_manager.get_scene_list()
            # Each scene is a tuple of (start_frame, end_frame).

            # Save stats if required.
            if stats_manager.is_save_required():
                with open(stats_file_path, 'w') as stats_file:
                    stats_manager.save_to_csv(stats_file)

        finally:
            # Release the video manager resources.
            video_manager.release()
        # print(scene_list)
        return scene_list
    
    def generate_description(self,video_path,as_string,byScene):
        
        scenes = self.find_scene_changes(video_path, method='content', new_stat_file=True)
        print(f'Scenes: {scenes}')
        scene_change_timecodes = [(scene[0].get_timecode(), scene[1].get_timecode())for scene in scenes]
        print(f"Scene change timecode: {scene_change_timecodes}")
        scene_change_idxs = [scene[0].get_frames() for scene in scenes]
        print(f"Scene change idxs:{scene_change_idxs}")

        if len(scene_change_idxs) == 0:
            print("No Scene Change!")
            scene_change_timecodes = ['00:00:00']
            scene_change_idxs = [0]
        else:
            print("Scene Change detected!")
        
        vid_embeddings = torch.zeros(
            len(scene_change_idxs), self.num_frames, self.vid_embedding_size)
        if torch.cuda.is_available():
            vid_embeddings = vid_embeddings.cuda()
        
        last_frame = scene_change_idxs[-1] + self.num_frames + 1

        frame_idx = 0
        cap_start_idx = 0
        
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()

            if not ret or frame_idx == last_frame:
                break

            # Start storing frames
            if frame_idx in scene_change_idxs:
                cap_start_idx = frame_idx
                vid_array = torch.zeros(self.num_frames, 3, 224, 224)

            # Transform, and store
            if frame_idx - cap_start_idx < self.num_frames:
                try:
                    frame = PIL.Image.fromarray(frame).convert('RGB')

                    if torch.cuda.is_available():
                        frame = self.transformer(frame).cuda().unsqueeze(0)
                    else:
                        frame = self.transformer(frame).unsqueeze(0)

                    vid_array[frame_idx - cap_start_idx] = frame

                except OSError as e:
                    print(e + " could not process frame in " + f)

            # If at scene ending frame, encode the collected scene
            if frame_idx - cap_start_idx == self.num_frames:
                if torch.cuda.is_available():
                    vid_array = vid_array.cuda()
                vid_embeddings[scene_change_idxs.index(
                    cap_start_idx)] =self.encoder(vid_array)

            frame_idx += 1

        cap.release()
        
        encoded_captions = self.video_captioner.predict(
        vid_embeddings, beam_size=5).cpu().numpy().astype(int)
        
        captions = []
        for caption in encoded_captions:
            captions.append(self.msrvtt_vocab.decode(
                caption, clean=True, join=True))
        
        return captions,scene_change_timecodes