#!/usr/bin/env python

from absl import app, flags
from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2 import run_configs
from pysc2.lib import actions, features, point
from os.path import basename, splitext
import os
from glob import glob
import numpy as np

import importlib

from Data import Dataline, State

FLAGS = flags.FLAGS
flags.DEFINE_string("replays", None, "Replay files pattern (google glob.glob)")
flags.DEFINE_string("datadir", "dataset", "Directory in which to put the data of the replays")
flags.DEFINE_bool("keepnoop", False, "Whether to keep no op actions.")
flags.mark_flag_as_required("replays")

class ReplayEnv:
    def __init__(self, replay_file_path, player_id=1, step_mul=1):
        self.replay_name = basename(splitext(replay_file_path)[0])
        self.step_mul = step_mul

        self.run_config = run_configs.get()
        self.sc2_proc = self.run_config.start()
        self.controller = self.sc2_proc.controller

        replay_data = self.run_config.replay_data(replay_file_path)
        ping = self.controller.ping()
        info = self.controller.replay_info(replay_data)
        if not self._valid_replay(info, ping):
            raise Exception("{} is not a valid replay file!".format(replay_file_path))

        interface = sc_pb.InterfaceOptions(raw=False, score=True, feature_layer=sc_pb.SpatialCameraSetup(width=24))
        point.Point(*Dataline.IMAGE_SHAPE).assign_to(interface.feature_layer.resolution)
        point.Point(*Dataline.IMAGE_SHAPE).assign_to(interface.feature_layer.minimap_resolution)

        map_data = None
        if info.local_map_path:
            map_data = self.run_config.map_data(info.local_map_path)

        self.controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id))

    @staticmethod
    def _valid_replay(info, ping):
        """Make sure the replay isn't corrupt, and is worth looking at."""
        return not (info.HasField("error") or
                    info.base_build != ping.base_build or  # different game version
                    info.game_duration_loops < 1000)

    def start(self):
        _features = features.Features(self.controller.game_info())
        states = []

        while True:
            obs = self.controller.observe()
            agent_obs = _features.transform_obs(obs.observation)

            # Assume that there is 0 or 1 action by frame.
            if obs.actions:
                states.append(State(agent_obs, _features.reverse_action(obs.actions[0])))
            elif FLAGS.keepnoop:
                states.append(State(agent_obs, actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])))

            if obs.player_result:
                break

            self.controller.step(self.step_mul)

        np.savez_compressed("{}/{}".format(FLAGS.datadir, self.replay_name), states=np.array(states))



def main(unused):
    replays = glob(FLAGS.replays)
    if not replays:
        print("No replays found.")
        return

    print("Replays found:")
    for r in replays:
        print(r)

    if not os.path.exists(FLAGS.datadir):
        print("\nCreating directory {}".format(FLAGS.datadir))
        os.mkdir(FLAGS.datadir)

    print("\nParsing...")
    for r in replays:
        print(r)
        env = ReplayEnv(r)
        env.start()

if __name__ == "__main__":
    app.run(main)
