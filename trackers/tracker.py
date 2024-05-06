import os
import pickle
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv


class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self,frames):
        batch_size = 20
        detections = []
        for i in tqdm(range(0,len(frames),batch_size)):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self,frames,read_tracks=False,tracks_path=None):
        
        if read_tracks and tracks_path is not None and os.path.exists(tracks_path):
            with open(tracks_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        # for each frame I have per detection(track_id) its position
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num,detection in tqdm(enumerate(detections)):

            class_names = detection.names
            class_index_inv = {v:k for k,v in class_names.items()} 

            # Now we convert to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert Goalkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = class_index_inv['player']

            # Track
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_index_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                if class_id == class_index_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                if class_id == class_index_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

            #for frame_detection in detection_supervision:
            #    bbox = frame_detection[0].tolist()
            #    class_id = frame_detection[3]
            #    if class_id == class_index_inv['ball']:
            #        tracks["ball"][frame_num][1] = {"bbox":bbox}

        if tracks_path is not None:
            with open(tracks_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id):
        y2 = int(bbox[3])
        

    def draw_tracking_ids(self,video_frames,tracks):
        output_frames = []
        for frame_num,frame in tqdm(enumerate(video_frames)):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball = tracks["ball"][frame_num]

            # Draw players trackers
            for track_id,track in player_dict.items():
                frame = self.draw_ellipse(frame,track["bbox"],(0,0,255),track_id)
