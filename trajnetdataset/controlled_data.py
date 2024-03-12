import argparse
import pickle
import random
import numpy as np
import rvo2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import os.path as osp
from tqdm import tqdm
from typing import List, Tuple

dest_dict = {}


def write_goals(self, out_file):
    if not osp.isdir('./goals'):
        os.makedirs('./goals')
    if not osp.isdir('./goals/train'):
        os.makedirs('./goals/train')
    with open(osp.join('./goals/train', out_file) + '.pkl', 'wb') as f:
        pickle.dump(dest_dict, f)

class Trajectory(object):
    def __init__(self, data: List[List[Tuple[float, float]]]):
        '''
        The Trajectory contain the trajectories of all the agent in a scene
        :param data: data
        '''
        self.data = data
        self.num_agents = len(data)

        assert self.num_agents > 0, "no agnet in this scene"


    def pop(self, i):
        self.data[i].pop(0)

    def append(self, i, position):
        self.data[i].append(position)

    def __getitem__(self, i):
        return self.data[i]

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

    def dump_csv(self, output_file, count, frame, dest_dict=None, goals = None):
        '''
        Write trajectories
        :param trajectories: Trajectory instance
        :param output_file:  output file path
        :param count: pedestrians count
        :param frame: frame limit
        :param dest_dict: the destination of pedestrian given ID as the key of the dict
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
        return last_frame

    def visualize(self, fps, show = True, save = False,
                  mode = None, video = True, out_file = None):
        '''
        visualize trajectories
        '''
        if not video:
            for i, trajectory in enumerate(self.data):
                trajectory = np.array(trajectory)
                plt.plot(trajectory[:,0], trajectory[:, 1], label=f'ped {i}')
            # plt.plot(self.data[:,0], self.data[:,1])
            plt.legend(loc='upper right')
            if mode == "trajnet":
                plt.xlim(-15, 15)
                plt.ylim(-15, 15)
            else:
                plt.xlim(-5, 5)
                plt.ylim(-5, 5)
            if show:
                plt.show()
                plt.close()
        if video:
            fig, ax = plt.subplots()
            if mode == "trajnet":
                ax.set_xlim(-15, 15)
                ax.set_ylim(-15, 15)
            else:
                ax.set_xlim(-5, 5)
                ax.set_ylim(-5, 5)


            premitive_list = [ax.plot([], [])[0] for _ in range(self.num_agents)]

            def init():
                [premitive_list[i].set_data([], []) for i in range(len(premitive_list))]
                return premitive_list

            def ani_func(frame):
                # print(frame)
                for i, trajectory in enumerate(self.data):
                    trajectory = np.array(trajectory)
                    x = trajectory[:frame+1, 0]
                    y = trajectory[:frame+1, 1]
                    # print(len(x), '\n')
                    # print(x)
                    premitive_list[i].set_xdata(x)
                    premitive_list[i].set_ydata(y)

                if frame == len(self.data[0]):
                    plt.close('all')

                return premitive_list

            ani = animation.FuncAnimation(fig, ani_func, init_func=init,
                                          frames=len(self.data[0])+1, interval=30,
                                          blit=False)
            # artists = []
            # for i in range(1, self.num_frame+1):
            #     artist = []
            #     for trajectory in self.data:
            #         trajectory = np.array(trajectory)
            #         artist.append(plt.plot(trajectory[:i, 0], trajectory[:i, 1])[0])
            #     artists.append(artist)
            # ani = animation.ArtistAnimation(fig = fig, artists=artists, interval=1 / fps, repeat=False, blit=False)
            if show:
                plt.show()
                # plt.close(fig)
            if save:
                writer = animation.FFMpegWriter(fps = fps, extra_args=['-vcodec', 'libx264'])
                ani.save(filename=out_file, writer=writer)







class Scene(object):

    def __init__(self, num_ped, name = "circle_crossing", sim = None, mode = "trajnet", seed = 42):
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

    # TODO: SFM trajectory generation
    def generate_sfm_trajectory(self, ):
        if self.name == "circle_crossing":
            pass
        else:
            raise NotImplementedError

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

    video_output_file = output_dir \
                        + args.simulator + '_' \
                        + args.simulation_scene + '_' \
                        + str(num_ped) + 'ped_' \
                        + str(num_scenes) + 'scenes_' \
                        + '.mp4'

    goal_filename = args.simulator + '_' \
                    + args.simulation_scene + '_' \
                    + str(num_ped) + 'ped_' \
                    + str(num_scenes) + 'scenes_'

    # overwrite the file generated before
    if osp.isfile(output_file):
        os.remove(output_file)

    count = 0  # pedestrian count
    last_frame = 0 # frame count
    dest_dict = {}  # path to store the description of dataset

    with tqdm(total=num_scenes) as pbar:
        for i in range(num_scenes):
            if mode == "trajnet":
                num_ped = random.choice([4, 5, 6])
            if i % 10 == 0:
                pbar.set_description(f'Episode {i+1}')
            # generation
            scene = Scene(num_ped)  # scene
            if args.simulator == "orca":
                trajectories, valid, goals = scene.generate_orca_trajectory(min_dist, react_time)

            elif args.simulator == 'social_force':
                pass

            else:
                raise NotImplementedError

            # Visualizing scenes
            if i % 45 == 0:
                trajectories.visualize(4, mode=scene.mode, out_file=video_output_file)

            # write the scene in csv format if valid
            if valid:
                last_frame = trajectories.dump_csv(output_file,
                                                   count=count,
                                                   frame=0,
                                                   dest_dict=dest_dict,
                                                   goals=goals)
            count += num_ped
            pbar.set_postfix({'valid': 'valid' if valid else 'invalid',
                              'ped': count,
                              'frame': last_frame
                              })
            pbar.update(1)

    write_goals(goal_filename)

    print(f'ORCA trajectories stored at: {output_file}')
    print(f'Goal information stored at: goal_files/train/{goal_filename}.pkl \n \n')

    print(f'You can convert this trajectories into TrajNet++ format using the following command \n')
    print(f'python -m trajnetdataset.convert --direct --synthetic --mode trajnet --linear_threshold 0.3 --acceptance 0.0 0.0 1.0 0.0 \
                --orca_file {output_file} --goal_file goal_files/train/{goal_filename}.pkl --output_filename orca_synthetic')

if __name__ == "__main__":
    main()