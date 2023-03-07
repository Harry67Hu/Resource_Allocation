import numpy as np
import random
import copy
from allocation_tasks.multiagent.core import World, Agent, Target
from allocation_tasks.multiagent.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment
from allocation_tasks.multiagent.basic_knowledge import Knowledge, ScenarioConfig


class Scenario(BaseScenario):
    def __init__(self, num_agents=4, dist_threshold=0.1, arena_size=1, identity_size=0, process_id=-1):

        self.num_agents = ScenarioConfig.num_agents
        self.knowledge = Knowledge(num_requirement_type=ScenarioConfig.num_requirement_type, num_plane_type=ScenarioConfig.num_plane_type, num_target_type=ScenarioConfig.num_target_type)
        self.joint_reward = 0

    def make_world(self):
        world = World()
        # 设置场景基本参数 e.g:
        world.max_episode_step = ScenarioConfig.max_episode_step
        world.num_requirement_type = ScenarioConfig.num_requirement_type
        world.num_plane_type = ScenarioConfig.num_plane_type
        world.num_target_type = ScenarioConfig.num_target_type
        world.max_num_plane = ScenarioConfig.max_num_plane

        
        # 定义基地智能体并赋予属性 
        world.agents = [Agent(num_plane_type=ScenarioConfig.num_plane_type, max_num_plane=ScenarioConfig.max_num_plane, num_requirement_type=ScenarioConfig.num_requirement_type) for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.index = i
            agent.state.real_planes = copy.deepcopy(self.knowledge.get_agent_plane()[i])
            agent.real_plane_threshold = copy.deepcopy(self.knowledge.get_agent_thres()[i])
        
        # 定义目标虚拟实体并赋予属性[因为目标数目不定，定义放到了reset_world函数中]

        self.reset_world(world)

        return world

    def reset_world(self, world):
        # 重置场景状态
        world.steps = 0
        world.done = False
        world.no_target2choose = False
        world.total_cost = 0

        # 重置智能体状态
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.index = i
            agent.state.planes = copy.deepcopy(self.knowledge.get_agent_plane()[i])
            agent.real_plane_threshold = copy.deepcopy(self.knowledge.get_agent_thres()[i])

        # 生成可行目标(可行解)
        a_int = random.randint(10, 29) # 第一个维度 a<30
        d_int = random.randint(20, 47) # 第二个维度 d<48
        e_int = random.randint(20, 41) # 第三个维度 e<42
        # a_int = random.randint(1, 29) # 第一个维度 a<30
        # d_int = random.randint(1, 47) # 第二个维度 d<48
        # e_int = random.randint(1, 41) # 第三个维度 e<42
        target_index = 0
        world.targets = [Target() for i in range(60 + 2*84 + 2*48 + 2*d_int)]
        for i in range(2*a_int):
            world.targets[target_index].type = 0
            world.targets[target_index].state.requirements = copy.deepcopy(self.knowledge.get_target_requirement()[0])
            target_index += 1
        for i in range(60 - 2*a_int):
            world.targets[target_index].type = 1
            world.targets[target_index].state.requirements = copy.deepcopy(self.knowledge.get_target_requirement()[1])
            target_index += 1
        for i in range(2*e_int):
            world.targets[target_index].type = 2
            world.targets[target_index].state.requirements = copy.deepcopy(self.knowledge.get_target_requirement()[2])
            target_index += 1
        for i in range(d_int + 3*d_int):
            world.targets[target_index].type = 3
            world.targets[target_index].state.requirements = copy.deepcopy(self.knowledge.get_target_requirement()[3])
            target_index += 1
        for i in range(2*(84-e_int)):
            world.targets[target_index].type = 4
            world.targets[target_index].state.requirements = copy.deepcopy(self.knowledge.get_target_requirement()[4])
            target_index += 1
        for i in range(2*(48-d_int)):
            world.targets[target_index].type = 5
            world.targets[target_index].state.requirements = copy.deepcopy(self.knowledge.get_target_requirement()[5])
            target_index += 1

        # 生成目标后对目标相关的量进行重置
        world.targets_type_remain = np.zeros(world.num_target_type)
        for i, target in enumerate(world.targets):
            target.index = i
            world.targets_type_remain[target.type] += 1
        world.max_episode_step = len(world.targets) * world.num_plane_type
        world.targets_done = np.zeros_like(world.targets,dtype=int)
        world.target_index_list = np.array([i for i,_ in enumerate(world.targets)])

        
        idx = np.random.choice(len(world.target_index_list))
        world.target_focus_on = world.target_index_list[idx]
        world.target_index_list = np.delete(world.target_index_list, idx)
        world.is_focus_done = False
        world.focus_time = 0

        assert True

    def reward(self, agent, world):
        '''
            智能体的奖励, 需要区分全体奖励和个体奖励(
            1. 当focus目标的需求被解决或者到达临时时间步后,给予奖励和成本惩罚(目前是只给奖励)
            2. 当episode结束后,全体智能体得到一个共同的奖励和成本惩罚
        '''
        self.joint_reward = 0
        single_reward = 0
        if world.done:
            done_partial = np.sum(world.targets_done) / len(world.targets_done)
            self.joint_reward += done_partial * ScenarioConfig.total_reward
            self.joint_reward -= world.total_cost
            # print("完成率为:{}".format(done_partial))
            # print("总成本为:{}".format(world.total_cost))
            return self.joint_reward
        else:
            if world.single_reward > 0:
                single_reward = world.single_reward * ScenarioConfig.single_target_reward
            return single_reward
    
    
    def observation(self, agent, world):
        '''
            基地智能体的观测, 为单个智能体的观测
            1. 自身的状态信息 (各类飞机剩余数目 + 基地位置？ )   约 10维
            2. 其他智能体的状态信息                           约 5*10维
            3. 当前主要目标的种类 (one-hot编码)               约 10维     
            4. 剩余目标种类的数目                            约 10维
            5. 当前目标的需求信息                            约 10维

            # 3. 当前主要目标的需求信息[固定需求] ps:固定需求指这一类目标本来的需求 【如果没有半Markov的话】
            # 4. 当前主要目标的需求信息[变化需求]
            # 3. 所有目标的需求信息(还是当前主要目标的需求信息 + 其他目标的需求信息？？)
            # 4. 所有目标的优先级信息？
        '''
        total_obs = None
        # part1
        part1 = agent.state.real_planes
        total_obs = part1
        # part2
        for ag in world.agents: 
            if ag.index != agent.index:
                total_obs = np.concatenate([total_obs, ag.state.real_planes])
        # part3 
        focus_target = world.targets[world.target_focus_on]
        type_code = np.zeros(world.num_target_type)
        type_code[focus_target.type] = 1
        total_obs = np.concatenate([total_obs, type_code])
        # part4 
        total_obs = np.concatenate([total_obs, world.targets_type_remain])
        # part5
        total_obs = np.concatenate([total_obs, focus_target.state.requirements])

        return total_obs

    def done(self, agent, world):
        '''
            episode终止条件
            1. 所有基地智能体到达能支出的阈值
            2. 超过最大时间步(所有目标都被遍历过且每个目标遍历的时候都把所有步长用了)(此情况不太常见)
            3. 所有目标都被遍历过但是没有成功
            4. 所有目标的需求被解决 = success
            
        '''
        condition1 = False
        condition2 = False
        condition3 = world.no_target2choose
        self.is_success = False

        # if world.num_is_agent_thres >= len(world.agents):
        #     condition1 = True
        if world.steps >= world.max_episode_step:
            condition2 = True
        if np.all(world.targets_done == 1):
            self.is_success = True
        
        world.done = condition1 or condition2 or condition3 or self.is_success
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
import cmath
import math
def convert_to_pole2D(vec):
    cn = complex(vec[0], vec[1])
    Range, rad_angle = cmath.polar(cn)
    angle = math.degrees(rad_angle)
    # print(Range, angle)    # (-180,+180]
    return Range, angle

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将经纬度转化为弧度
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # 地球平均半径，单位为千米
    return c * r