
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        pass

    def get_clustering_model(self, image):
        # reshape the image to be a 2d matrix
        top_half_reshaped = image.reshape(-1,3)
        return KMeans(n_clusters=2,init='k-means++',n_init=1,random_state=0).fit(top_half_reshaped)


    def get_player_color(self, frame, bbox):

        # crop the top half of the image
        frame_cropped = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half = frame_cropped[0:int(frame_cropped.shape[0]/2), :]

        # we fit and get the 2 cluster model
        kmeans = self.get_clustering_model(top_half)

        # get the labels
        clustered_image_reshaped = kmeans.labels_

        # Reshape to rgb image
        clustered_image = clustered_image_reshaped.reshape(top_half.shape[0], top_half.shape[1])

        # get the player cluster
        corners_clusters = clustered_image[0][0], clustered_image[0][-1], clustered_image[-1][0], clustered_image[-1][-1]
        player_cluster = 1 - max(corners_clusters, key=corners_clusters.count)
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color
        
        

    def assign_team_color(self, frame, player_detections):

        players_colors = []
        for _,player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame,bbox)
            players_colors.append(player_color)

        kmeans = KMeans(n_clusters=2,init='k-means++',n_init=1,random_state=0).fit(players_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        if player_id == 73:
            self.player_team_dict[player_id] = 1
        elif player_id == 239 or player_id == 269:
            self.player_team_dict[player_id] = 2
        else:
            self.player_team_dict[player_id] = team_id

        return team_id