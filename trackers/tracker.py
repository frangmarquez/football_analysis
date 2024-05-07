import os
import pickle
import sys
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width


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
                

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                if class_id == class_index_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if tracks_path is not None:
            with open(tracks_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame,
                    center=(x_center, y2),
                    axes=( int(0.8*width), int(0.25 * width)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4
                    )
        
        rectangle_width = 40
        rectangle_height = 20
        x1rectangle = int(x_center - rectangle_width/2)
        y1rectangle = int(y2 + rectangle_height/2 +15)
        x2rectangle = int(x_center + rectangle_width/2)
        y2rectangle = int(y2 - rectangle_height/2 +15)

        if track_id is not None:
            cv2.rectangle(frame,
                          (x1rectangle, y1rectangle),
                          (x2rectangle, y2rectangle),
                          color,
                          cv2.FILLED)
            x1_text = int(x1rectangle + 12)
            if track_id > 99:
                x1_text -= 10
            cv2.putText(frame,
                        str(track_id),
                        (int(x1_text),int(y2rectangle+15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA)
        return frame
    
    def draw_triangle(self,frame,bbox,color):
        y1 = int(bbox[1])
        x_center, _ = get_center_of_bbox(bbox)

        cv2.drawContours(frame,
                         [np.array([[x_center, y1],[x_center-10,y1-20],[x_center+10,y1-20]])],
                         0,
                         color,
                         cv2.FILLED)
        cv2.drawContours(frame,
                         [np.array([[x_center, y1],[x_center-10,y1-20],[x_center+10,y1-20]])],
                         0,
                         (0,0,0),
                         2)
        
        return frame


    def draw_tracking_ids(self,video_frames,tracks):
        output_frames = []
        for frame_num,frame in tqdm(enumerate(video_frames)):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball = tracks["ball"][frame_num]

            # Draw trackers
            for track_id,track in player_dict.items():
                frame = self.draw_ellipse(frame,track["bbox"],track["team_color"],track_id)

            for _, track in referee_dict.items():
                frame = self.draw_ellipse(frame,track["bbox"],(0,255,255))

            for _, track in ball.items():
                frame = self.draw_triangle(frame,track["bbox"],(0,255,0))

            output_frames.append(frame)
        
        return output_frames
