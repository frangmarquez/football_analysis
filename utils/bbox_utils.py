import math

def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def measure_xy_movement(new_feature_point, old_feature_point):
        camera_movement_x = new_feature_point[0] - old_feature_point[0]
        camera_movement_y = new_feature_point[1] - old_feature_point[1]
        return camera_movement_x, camera_movement_y