import argparse
import random
import numpy as np
import rvo2
import matplotlib.pyplot as plt
import os
import os.path as osp
from tqdm import tqdm
from typing import List, Tuple


class Trajectory(object):
    def __init__(self, data: List[List[Tuple[float, float]]]):
        '''
        The Trajectory contain the trajectories of all the agent in a scene
        :param data: data
        '''
        self.data = data

    def pop(self, i):
        self.data[i].pop(0)

    def append(self, i, position):
        self.data[i].append(position)

    def are_smoothes(self):
        '''
        Check if there is no sharp turns in the trajectories
        '''
        def get_angle(a, b, c):
            '''
                 a
                /
              /
            /
           b------c
           compute the angle between ba and bc
            '''
            ba = a - b
            bc = c - b
            cos_angle_rad = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            return np.arccos(cos_angle_rad)

        is_smooth = True
        for i, trajectory in enumerate(self.data):
            for j in range(len(trajectory) - 3):
                p1 = np.array(trajectory[j])
                p2 = np.array(trajectory[j+1])
                p3 = np.array(trajectory[j+2])
                angle = get_angle(p1, p2, p3)
                if angle < np.pi / 2:
                    is_smooth = False
                    plt.scatter(p1[0], p1[1], color='red', marker='X')
        return is_smooth

    def dump_csv(self, output_file, count, frame, dest_dict = None, goals = None):
        '''
        Write trajectories
        :param trajectories: Trajectory instance
        :param output_file:  output file path
        :param count: pedestrians count
        :param frame: frame limit
        :param dest_dict:
        :param goals: goals of the pedestrians
        :return: the last frame number
        '''
        last_frame = 0
        with open(output_file, 'a') as f:
            track_data = []
            for i, trajectory in enumerate(self.data):
                for t,point in enumerate(trajectory):
                    track_data.append(f'{t+frame},{i+count},{point[0]},{point[1]}')
                    if t == len(trajectory)-1 and t+frame > last_frame:
                        last_frame = t+frame
                if goals:
                    dest_dict[count+i] = goals[i]
            for track in track_data:
                f.write(track)
                f.write('\n')



class Scene(object):

    def __init__(self, num_ped, name = "circle_crossing", sim = None, mode = None, seed = 42):
        # store the arguments
        self.sim = sim
        self.name = name
        self.num_ped = num_ped
        self.mode = mode
        self.seed = seed

    def generate_trajectories(self):
        '''
        The function for generating trajectories for this scene
        :return:
        '''
        pass

    def generate_orca_trajectory(self, min_dist = 3., react_time = 1.5, end_range = 1.0):
        '''
        generate trajectories using ORCA
        :param min_dist: minimal distance for each pair of pedestrians during simulation
        :param react_time:
        :param end_range: the distance between the agent and its goal to stop the agent
        :return: trajectory, valid, goals
        '''
        if self.name == 'circle_crossing':
            fps = 100
            sampling_rate = fps / 2.5
            self.sim = rvo2.PyRVOSimulator(1/fps, 10, 10, 5, 5, 0.3, 1)
            if self.mode == 'trajnet':
                self.sim = rvo2.PyRVOSimulator(1/fps, 4, 10, 4, 5, 0.6, 1.5)
            trajectories, _, goals, speed = self.generate_circle_crossing()
        else:
            raise NotImplementedError

        done = False
        reaching_goal_by_ped = [False] * self.num_ped
        count = 0
        valid = True
        while not done and count < 6000:
            count += 1
            self.sim.doStep()
            reaching_goal = []
            for i in range(self.num_ped):
                if count == 1:
                    trajectories.pop(i)
                position = self.sim.getAgentPosition(i)

                ## Append only if Goal not reached
                if not reaching_goal_by_ped[i]:
                    if (count-1) % sampling_rate == 0:
                        trajectories.append(i, position)

                # check if this agent reaches the goal
                if np.linalg.norm(np.array(position) - np.array(goals[i])) < end_range:
                    reaching_goal.append(True)
                    self.sim.setAgentPrefVelocity(i, (0, 0))
                    reaching_goal_by_ped[i] = True
                else:
                    reaching_goal.append(False)
                    velocity = np.array((goals[i][0] - position[0], goals[i][1] - position[1]))
                    speed = np.linalg.norm(velocity)
                    pref_vel = 1 * velocity / speed if speed > 1 else velocity
                    self.sim.setAgentPrefVelocity(i, tuple(pref_vel.tolist()))
            done = all(reaching_goal)

        # validate the trajectory
        if not done or not trajectories.are_smoothes():
            valid = False

        return trajectories, valid, goals


    def generate_circle_crossing(self, radius = 4):
        '''
        generate the initial state of circle crossing settings
        :param radius: the radius of the circle for circle crossing
        :return: trajectories(only initial point), points, initial speed, goals
        '''
        positions = []
        goals = []
        speed = []
        agent_list = [] # state of agents
        if self.mode == "trajnet":
            radius = 10
        for _ in range(self.num_ped):
            while True:
                angle = random.uniform(0, 1) * 2 * np.pi
                # add some noise
                px_noise = random.uniform(0, 1) - 0.5
                py_noise = random.uniform(0, 1) - 0.5
                px = radius * np.cos(angle) + px_noise
                py = radius * np.sin(angle) + py_noise
                collide = False
                for agent in agent_list:
                    min_dist = 0.8
                    if self.mode == "trajnet":
                        min_dist = 2
                    # start position must not be the same or on the opposite
                    if np.linalg.norm((px - agent[0], py - agent[1])) < min_dist or \
                        np.linalg.norm((px - agent[2], py - agent[3])) < min_dist:
                        collide = True
                        break
                if not collide:
                    break

            positions.append((px, py))
            goals.append((-px, -py))
            if self.sim is not None:
                self.sim.addAgent((px, py))
            velocity = np.array([-2 * px, -2 * py])
            magnitude = np.linalg.norm(velocity)
            init_vel = 1 * velocity / magnitude if magnitude > 1 else velocity
            speed.append([init_vel[0], init_vel[1]])
            agent_list.append([px, py, -px, -py])
        trajectories = Trajectory([[positions[i]] for i in range(self.num_ped)])
        return trajectories, positions, goals, speed

    # TODO other scene topologies



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator', default='orca',
                        choices=('orca', 'social_force'))
    parser.add_argument('--simulation_scene', default='circle_crossing',
                        choices=('circle_crossing'))
    parser.add_argument('--mode', default=None,
                        help='Set to trajnet for trajnet-based dataset generation')
    parser.add_argument('--num_ped', type=int, default=6,
                        help='Number of ped in scene, if mode=trajnet then num_ped is randomly chosen from (4, 5, 6)')
    parser.add_argument('--num_scenes', type=int, default=100,
                        help='Number of scenes')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.seterr('ignore')
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Decide the number of scenes & agents per scene
    num_scenes = args.num_scenes
    num_ped = args.num_ped
    mode = args.mode
    min_dist, react_time = 1.5, 1.5

    if not osp.isdir("./data"):
        os.makedirs("./data")
    if not osp.isdir("./data/raw"):
        os.makedirs("./data/raw")
    if not osp.isdir("./data/raw/synthesized"):
        os.makedirs("./data/raw/synthesized")

    # files to write the scene
    output_dir = "data/raw/synthesized/"
    output_file = output_dir \
                  + args.simulator + '_' \
                  + args.simulation_scene + '_' \
                  + str(num_ped) + 'ped_' \
                  + str(num_scenes) + 'scenes_' \
                  + '.txt'

    goal_filename = args.simulator + '_' \
                    + args.simulation_scene + '_' \
                    + str(num_ped) + 'ped_' \
                    + str(num_scenes) + 'scenes_'

    # overwrite the file generated before
    if osp.isfile(output_file):
        os.remove(output_file)

    count = 0  # pedestrian count
    last_frame = -5 # frame count
    dest_dict = {}  # path to store the description of dataset


    with tqdm(total=num_scenes) as pbar:
        for i in range(num_scenes):
            if mode == "trajnet":
                num_ped = random.choice([4, 5, 6])
            if i % 10 == 0:
                pbar.set_description(f'round {i+1} simulation')
            # generation
            scene = Scene(num_ped)  # scene
            if args.simulator == "orca":
                trajectories, valid, goals = scene.generate_orca_trajectory(min_dist, react_time)
            # TODO SFM

            elif args.simulator == 'social_force':
                pass

            else:
                raise NotImplementedError

            # TODO visualizing the scene
            # Visualizing scenes

            # write the scene in csv format if valid
            if valid:
                last_frame = trajectories.dump_csv(output_file,
                                                   count=count,
                                                   frame=0,
                                                   dest_dict=dest_dict,
                                                   goals=goals)
            count += num_ped
            pbar.set_postfix({'valid or not': 'valid' if valid else 'invalid'})
            pbar.update(1)

if __name__ == "__main__":
    main()