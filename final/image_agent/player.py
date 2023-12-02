import os
import numpy as np
import torch
from .detector import Detector, CNNClassifier


class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = Detector().to(self.device)
        if os.path.exists("image_agent/det.th"):
            print("Loading detector model...")
            self.detector.load_state_dict(torch.load("image_agent/det.th", map_location="cpu"))
            print("Done!")
        self.classifier = CNNClassifier().to(self.device)
        if os.path.exists("image_agent/cnn.th"):
            print("Loading classifier model...")
            self.classifier.load_state_dict(torch.load("image_agent/cnn.th", map_location="cpu"))
            print("Done!")
        self.team = None
        self.num_players = None

        # state memory
        self.last_loc = [[0, 0], [0, 0]]
        # bools to track unstucking behavior
        self.getting_out_of_goalpost = [False, False]
        self.getting_off_of_wall = [False, False]
        # frame counters
        self.frame = 0
        self.unstucking_frames = [0, 0]

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """

        action_dicts = []
        self.frame += 1
        for i in range(self.num_players):
            # print(f"PLAYER {i}")

            # calculating various values
            img = torch.tensor(np.transpose(player_image[i], [2, 0, 1]), dtype=torch.float).to(self.device)
            puck_coords = self.detector.detect(img)
            puck_x = float(puck_coords[0].item())
            puck_y = float(puck_coords[1].item())
            classify_output = float(self.classifier.forward(img[None])[0].item())
            seeing_puck = classify_output > 0.5
            print(round(classify_output, 2) , "->", seeing_puck)
            dir_vec = np.array(player_state[i]["kart"]["front"]) - np.array(player_state[i]["kart"]["location"])
            loc_change = ((player_state[i]["kart"]["location"][0] - self.last_loc[i][0])**2 + (player_state[i]["kart"]["location"][2] - self.last_loc[i][1])**2)**0.5
            if loc_change > 10:
                self.frame = 1

            # setting values for normal behavior (may be changed by later code for edge cases)
            if np.linalg.norm(player_state[i]["kart"]["velocity"]) < 12:
                    acceleration = 0.5
            elif np.linalg.norm(player_state[i]["kart"]["velocity"]) > 15:
                acceleration = 0
                brake = True
            else:
                acceleration = 0
            brake = False
            steer = 0

            # print(f"position of {i}:", player_state[i]["kart"]["location"])
            # print(f"direction of {i}:", dir_vec)

            # detecting if we need to get unstuck
            in_goalpost = abs(player_state[i]["kart"]["location"][2]) > 64
            stuck_against_x_dir_wall = (
                abs(player_state[i]["kart"]["location"][0]) > 40  # located next to x wall
                and abs(dir_vec[0]) > abs(dir_vec[2])  # primarily pointing in x dir
                and np.sign(player_state[i]["kart"]["location"][0]) == np.sign(dir_vec[0])  # pointed towards wall
            )
            stuck_against_y_dir_wall = (
                abs(player_state[i]["kart"]["location"][2]) > 63  # located next to y wall
                and abs(dir_vec[2]) > abs(dir_vec[0])  # primarily pointing in y dir
                and np.sign(player_state[i]["kart"]["location"][2]) == np.sign(dir_vec[2])  # pointed towards wall
            )
            # print(f"Player {i}:", in_goalpost, stuck_against_x_dir_wall, stuck_against_y_dir_wall)

            # rush the puck in the beginning of the game
            if self.frame <= 60:
                acceleration = 1
                steer = 0

            # get out of goalpost if stuck in it
            elif in_goalpost or self.getting_out_of_goalpost[i]:
                # print(f"Player {i} escaping goalpost")
                self.getting_out_of_goalpost[i] = True
                # back up in straight line
                if self.unstucking_frames[i] < 40:
                    acceleration = 0
                    brake = True
                    steer = 0
                    self.unstucking_frames[i] += 1
                # turn as hard as you can
                elif self.unstucking_frames[i] < 60:
                    acceleration = 1
                    steer = 1
                    self.unstucking_frames[i] += 1
                else:
                    self.getting_out_of_goalpost[i] = False
                    self.unstucking_frames[i] = 0

            # get off of wall if stuck against it
            elif stuck_against_x_dir_wall or stuck_against_y_dir_wall or self.getting_off_of_wall[i]:
                # print(f"Player {i} getting off wall")
                self.getting_off_of_wall[i] = True
                # back up in straight line
                if self.unstucking_frames[i] < 20:
                    acceleration = 0
                    brake = True
                    steer = 0
                    self.unstucking_frames[i] += 1
                # turn as hard as you can
                elif self.unstucking_frames[i] < 40:
                    acceleration = 1
                    steer = 1
                    self.unstucking_frames[i] += 1
                else:
                    self.getting_off_of_wall[i] = False
                    self.unstucking_frames[i] = 0

            # GO AFTER THAT PUCK!!
            elif seeing_puck:
                acceleration = 1
                steer = puck_x

            # Find the puck quickly
            else:
                # we are facing away from the center of the arena
                if np.dot(dir_vec, player_state[i]["kart"]["location"]) / (np.linalg.norm(dir_vec) * np.linalg.norm(player_state[i]["kart"]["location"])) > 0.2:
                    steer = 1

            action = dict(
                acceleration=acceleration,
                brake=brake,
                drift=abs(steer) > 0.7,
                fire=False,
                nitro=False,
                rescue=False,
                steer=steer
            )
            action_dicts += [action]

            self.last_loc[i][0] = player_state[i]["kart"]["location"][0]
            self.last_loc[i][1] = player_state[i]["kart"]["location"][2]
        return action_dicts
