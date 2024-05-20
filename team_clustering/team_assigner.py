from sklearn.cluster import KMeans


class TeamAssigner:

    __GOALKEEPER_TEAM1 = 91  # hard-coded goalkeeper player ID
    __GOALKEEPER_TEAM2 = 92  # hard-coded goalkeeper player ID

    def __init__(self):
        self.team_colours = {}
        self.player_team_dict = {}      # {player_id: team_id} format
        self.kmeans = None

    def get_clustering_model(self, image):
        # Reshape the image into an array:
        image_2d = image.reshape(-1, 3)

        # Perform k-means with 2 clusters:
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init="auto").fit(image_2d)

        return kmeans

    def get_player_colour(self, frame, bbox):
        # Cropping the image, only taking everything inside the bounding box:
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        image_top_half = image[0:int(image.shape[0]/2), :]

        # Getting the k-means clustering model:
        kmeans = self.get_clustering_model(image=image_top_half)

        # Get the cluster labels for each "pixel type":
        labels = kmeans.labels_

        # Reshape the labels to the image shape:
        clustered_image = labels.reshape(image_top_half.shape[0], image_top_half.shape[1])

        # Get the player cluster:
        #   corner_clusters:    the corner of the image, most likely background cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1],
                           clustered_image[-1, 0], clustered_image[-1, -1]]
        #   non_player_cluster: the maximum "type" of cluster out of the corner clusters
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        #   player_cluster:     the opposite cluster to the non_player_cluster
        player_cluster = 1 - non_player_cluster

        # Now we get the player colour by taking the average colour of the player_cluster:
        player_colour = kmeans.cluster_centers_[player_cluster]

        return player_colour

    # Limitations: cannot calculate team of the goalkeeper.
    def assign_team_colour(self, frame, player_detections):
        # For each player, we put the colour into a list:
        player_colours = []

        # Iterating over all the player_detections:
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_colour = self.get_player_colour(frame, bbox)
            player_colours.append(player_colour)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init="auto")
        kmeans.fit(player_colours)

        self.kmeans = kmeans

        self.team_colours[1] = kmeans.cluster_centers_[0]
        self.team_colours[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        # If we have already calculated this player's team:
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Calculating the player colour given the frame and bbox:
        player_colour = self.get_player_colour(frame, player_bbox)

        # Getting the team_id for this player (notice team_id will be 0 or 1, but we want 1 or 2)
        team_id = self.kmeans.predict(player_colour.reshape(1, -1))[0]
        team_id += 1

        if player_id == self.__GOALKEEPER_TEAM1:
            team_id = 1
        elif player_id == self.__GOALKEEPER_TEAM2:
            team_id = 2

        self.player_team_dict[player_id] = team_id

        return team_id
