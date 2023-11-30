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
        if os.path.exists("homework/planner.th"):
            self.model.load_state_dict(torch.load("homework/planner.th"))
        self.team = None
        self.num_players = None

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
            img = torch.tensor(np.transpose(player_image[i], [2, 1, 0]), dtype=torch.float).to(self.device)
            puck_locations = self.model.forward(img[None])
            puck_x = float(puck_locations[0][0].item())
            steering = 0
            drifting = False
            vel = np.linalg.norm(player_state[i].get('kart').get('velocity'))
            print(player_state[i].get('kart').get('location'))
            acc = 1
            if (puck_x <0) :
                steering = -1
            if (puck_x > 0):
                steering = 1
            if (abs(puck_x) > 0.3):
              drifting = True
            if (vel > 5):
              acc = 0
            # making default action for now
            action = dict(
                acceleration=acc,
                brake=False,
                drift=drifting,
                fire=False,
                nitro=False,
                rescue=False,
                steer=steering
            )
            action_dicts += [action]
        return action_dicts
