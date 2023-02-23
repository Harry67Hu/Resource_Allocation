import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment


class ScenarioConfig(object):  
    '''
        场景参数放置于此处
    '''
    # e.g:
    num_agents = 9
    num_landmarks = 3
    dist_threshold = 0.1
    arena_size = 1

class Scenario(BaseScenario):
    def __init__(self, num_agents=4, dist_threshold=0.1, arena_size=1, identity_size=0, process_id=-1):
        # e.g:
        self.num_agents = ScenarioConfig.num_agents
        self.num_landmarks = ScenarioConfig.num_landmarks
        self.arena_size = ScenarioConfig.arena_size
        self.dist_thres = ScenarioConfig.dist_threshold

    def make_world(self):
        world = World()
        # 设置场景基本参数 e.g:
        world.dim_c = 2
        world.collaborative = False
        
        # 定义基地智能体并赋予属性
        world.agents = [Agent(iden=i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
        
        # 定义目标虚拟实体并赋予属性
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i


        self.reset_world(world)

        return world

    def reset_world(self, world):
        # 重置场景状态
        world.steps = 0


    def reward(self, agent, world):
        '''
            智能体的奖励, 需要区分全体奖励和个体奖励
            1. 每一步所有智能体的资源贡献使得当前步的目标需求被解决，得到奖励
            2. episode结束的时候根据所有目标的毁伤情况给一个奖励
            3. episode结束的时候给所有智能体给予一定的成本惩罚(整体)
        '''
        joint_reward = 0
        if world.done:
            joint_reward = joint_reward + 50 if np.all(world.targets_done == 1) else joint_reward + 0
            joint_reward -= world.cost
        else:
            main_target = world.targets[world.steps]
            requirements = main_target.state.requirements
            if np.all(requirements == 0):
                print("目标{}的需求被解决".format(world.steps))
                joint_reward += 1

        return joint_reward
        return single_reward
    
    
    
    def observation(self, agent, world):
        '''
            基地智能体的观测, 为单个智能体的观测
            1. 自身的状态信息 (基地位置？ + 各类飞机剩余数目)
            2. 其他智能体的状态信息
            3. 所有目标的需求信息(还是当前主要目标的需求信息 + 其他目标的需求信息？？)
            4. 所有目标的优先级信息？
        '''
        total_obs = None
        # part1
        part1 = agent.state.planes
        total_obs = part1
        # part2
        for ag in world.agents: 
            if ag.index != agent.index:
                total_obs = np.concatenate(total_obs, ag.state.planes)
        # part3 
        for target in world.targets:
            total_obs = np.concatenate(total_obs, target.state.requirements)
        # part3.5 当前指标（指向要解决的目标索引）

        # part4 所有目标的优先级信息？？ 

        return total_obs

    def done(self, agent, world):
        '''
            episode终止条件
            1. 所有基地智能体到达能支出的阈值
            2. 超过最大时间步
            3. 所有目标的需求被解决 = success
            
        '''
        condition1 = False
        condition2 = False
        self.is_success = False

        if world.num_is_agent_thres >= len(world.agents):
            condition1 = True
        if world.steps >= world.MaxEpisodeStep:
            condition2 = True
        if np.all(world.targets_done == 1):
            self.is_success = True
        
        world.done = condition1 or condition2 or self.is_success
        return world.done

    def info(self, agent, world):
        '''
            存储实验结果以供参考
        '''

        return {'is_success': self.is_success, 'world_steps': world.steps,
                'reward':self.joint_reward, }
    


'''
计算用辅助函数部分
'''

def get_thetas(poses):
    # compute angle (0,2pi) from horizontal
    thetas = [None]*len(poses)
    for i in range(len(poses)):
        # (y,x)
        thetas[i] = find_angle(poses[i])
    return thetas

def find_angle(pose):
    # compute angle from horizontal
    angle = np.arctan2(pose[1], pose[0])
    if angle<0:
        angle += 2*np.pi
    return angle

def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        min_dists = dists[ri, ci]
        return min_dists

def Norm(x):  
    return np.linalg.norm(x)

def convert_to_pole2D(vec):
    cn = complex(vec[0], vec[1])
    Range, rad_angle = cmath.polar(cn)
    angle = math.degrees(rad_angle)
    # print(Range, angle)    # (-180,+180]
    return Range, angle