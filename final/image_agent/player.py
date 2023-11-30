import os
import numpy as np
import torch
from .detector import Detector


class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Detector().to(self.device)
        if os.path.exists("image_agent/det.th"):
            print("Loading saved model...")
            self.model.load_state_dict(torch.load("image_agent/det.th"))
            print("Done!")
        self.team = None
        self.num_players = None

        # counters to help with getting unstuck
        self.backup_frames = [0, 0]
        self.turn_frames = [0, 0]

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
        for i in range(self.num_players):
            # calculating puck's x and y coordinates on screen
            img = torch.tensor(np.transpose(player_image[i], [2, 1, 0]), dtype=torch.float).to(self.device)
            puck_coords = self.model.forward(img[None])[0]
            puck_x = float(puck_coords[0].item())
            puck_y = float(puck_coords[1].item())

            # setting values for normal behavior (may be changed by later code for edge cases)
            if np.linalg.norm(player_state[i]["kart"]["velocity"]) < 15:
                acceleration = 1
            else:
                acceleration = 0
            brake = False
            steer = puck_x

            # don't get stuck in a goalpost
            if abs(player_state[i]["kart"]["location"][2]) > 64 or self.backup_frames[i] > 0:
                print(f"Player {i} escaping goalpost")
                print("position:", player_state[i]["kart"]["location"])
                print("direction:", player_state[i]["kart"]["front"] - player_state[i]["kart"]["location"])
                if self.backup_frames[i] < 25:
                    acceleration = 0
                    brake = True
                    steer = 0
                    self.backup_frames[i] += 1
                elif self.turn_frames[i] < 40:
                    steer = 1
                    self.turn_frames[i] += 1
                else:
                    self.backup_frames[i] = 0
                    self.turn_frames[i] = 0

            # don't get stuck against the wall
            elif abs(player_state[i]["kart"]["location"][0]) > 35 or player_state[i]["kart"]["location"][2] > 64:
                print(f"Player {i} getting off wall")
                print("position:", player_state[i]["kart"]["location"])
                print("direction:", player_state[i]["kart"]["front"] - player_state[i]["kart"]["location"])

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
        return action_dicts
