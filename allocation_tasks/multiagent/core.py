import numpy as np
from random import sample
from .basic_knowledge import Knowledge, ScenarioConfig
# State
class EntityState(object):
    def __init__(self):
        '''
            此处定义实体模型的基本状态
            使用样例 e.g :entity.state.p_vel = entity.state.p_vel / speed * entity.max_speed
        '''
        self.is_alive = None

class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        '''
            此处定义基地智能体的基本状态 (可以跟随动作变化)
            1. 基地内各类飞机剩余数目
            2. 基地内各类载荷剩余数目(可能有）
        '''

        self.real_planes = None # 各类飞机挂载剩余数目列表，在scenario中进行初始化

class TargetState(EntityState):
    def __init__(self):
        super(TargetState, self).__init__()
        '''
            此处定义目标的基本状态 (可以跟随动作变化)
            1. 目标需求
            2. 目标优先级
        '''
        
        self.requirements = None # 目标需求列表,可以在在scenario中进行初始化,也可以通过type进行加载
        self.priority = None


# Action 
class Action(object):
    def __init__(self, num_plane_type=12, max_num_plane=4, num_requirement_type=7):
        '''
            此处定义基地智能体的动作
            使用样例 e.g : state[t+1] = state[t] + agent.action.example 
            【方案一】在这个方案里面, 每个action在一步中只对一个目标产生作用
            
            0. 如果智能体在一步内对某个目标进行决策，则其离散动作数目为 (max_num_plane + 1)^num_plane_type 这是不可接受的！
            TODO  半马尔科夫模型？？？ 【即智能体在多步对一个目标决策后，目标的状态才进行变化？？】
            1. 智能体的动作为选定特定类型的飞机并决定多少架 + 空动作 num_plane_type * (max_num_plane + 1) + 1空动作
            2. 分配资源的形式为 len = num_plane_type  e.g. 0号动作为 real_act = [0,0,0,0,0,0,0,0,0,0,0,0]   act act = 29号动作为 real_act = [2,0,0,0,0,0,0,0,0,0,0,0] 30号动作为[3,0,0,0,0,0,0,0,0,0,0,0]
            3. 对智能体的影响为 agent_state -= real_act
            4. 对目标的影响需要根据飞机的载荷实际计算目标的减少需求
        '''
        self.num_plane_type = num_plane_type
        self.max_num_plane = max_num_plane
        self.num_requirement_type = num_requirement_type
        self.act = None # 智能体的动作,为一个数,此为强化学习中真正使用的动作！

    # 此处将数字act转译为真实act
    def act2real(self):
        if self.act == 0:
            real_act = np.zeros(self.num_plane_type, dtype=int) 
        else:
            axis = (self.act -1) // self.max_num_plane
            num = (self.act -1) % self.max_num_plane + 1
            real_act = np.zeros(self.num_plane_type, dtype=int) 
            real_act[axis] = num
        return real_act
    # 此处将数字act转译为分配出来的载荷
    def act2source(self):
        real_act = self.act2real()
        knowledge = Knowledge(num_requirement_type=self.num_requirement_type, num_plane_type=self.num_plane_type, num_target_type=self.num_target_type)
        PLANE_CAPACITY = knowledge.get_plane_capacity()
        resource_output = np.zeros(self.num_requirement_type, dtype=int)
        for i, num in enumerate(real_act):
            resource_output += num * PLANE_CAPACITY[i]
        return resource_output


# Entity Model
class Entity(object):
    def __init__(self):
        '''
            此处定义可以调用的实体模型
        '''
        self.state = EntityState()
        # name 
        self.name = ''
        # properties:
        self.size = None
        self.location = None
        

class Target(Entity):
     def __init__(self):
        super(Target, self).__init__()
        '''
            此处定义目标模型的基本属性
            基本状态以外的模型和参数
            1. 目标位置
            2. 目标不确定性因素 TODO ?
        '''
        # state
        self.state = TargetState()
        # e.g: 
        self.color = None
        self.type = 0 # 目标种类
        self.index = 0

class Agent(Entity):
    def __init__(self, num_plane_type=0, max_num_plane=0, num_requirement_type=0):
        super(Agent, self).__init__()
        '''
            此处定义基地智能体模型的基本属性
            基本状态以外的模型和参数
            1. 基地位置
            2. 基地能派出飞机的阈值(此处的类型是真飞机种类)
        '''
        # state
        self.state = AgentState()
        # action
        self.action = Action(num_plane_type=num_plane_type, max_num_plane=max_num_plane, num_requirement_type=num_requirement_type)
        self.action_callback = None
        # properties
        self.index = 0 # 基地智能体的索引
        self.real_plane_threshold = None # e.g. [36,36,0,0,0]

        

# World Model with entity model 
class World(object):
    def __init__(self):

        self.num_requirement_type = 0
        self.num_plane_type = 0      # 飞机挂载的种类
        self.num_target_type = 0

        self.max_num_plane = 36 # 每个机场每类飞机最大的调用数目

        self.agents = []
        self.targets = []
        self.targets_done = [] # 存储每个目标是否被解决的向量,初始全部为0,被解决则为1
        self.num_is_agent_thres = 0 # 存储到达阈值的基地数目

        self.steps = 0
        self.max_episode_step = 50
        self.done = False # 整个episode结束

        self.target_focus_on = 0 # 当前步集中考虑的目标索引
        self.target_index_list = [] # 当前步剩余的还没有考虑的目标索引列表
        self.is_focus_done = False # 当前集中考虑的目标是否被解决
        self.focus_time = 0 # 此时间超过最大后重置，并更新当前集中考虑的目标

        self.targets_type_remain = np.zeros(self.num_target_type) # 剩余各种类目标的数目，仅观测使用

        self.total_cost = 0 # 所有资源对应的成本，供episode结束使用，仅奖励使用
        self.single_cost = 0 # 当前为解决focus目标所有资源对应的成本，仅奖励使用
        self.single_reward = 0 # 当前为解决focus目标给予的奖励，仅奖励使用

        self.collaborative = True
    # debug使用
    @property
    def entities(self):
        return self.agents + self.targets

    # 调用基于强化学习的基地智能体
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # 给运筹等其他算法开放的接口智能体
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self):
        '''
            此处为环境的基本运行规则   (有些部分虽然与reward相关, 但是放到规则部分, reward负责直接调用)
            0. 环境基本状态转移模型
                0.1. 当目标的需求被解决或者在子任务中step超过num_plane_type的时候,focus目标更改
                0.1. 每一步中分别对每个智能体的剩余飞机数目和当前目标的需求做更新
            1. 载荷及飞行成本模型（包括考虑每个基地是否达到其阈值)
            2. 作战效益模型【暂时没加进去】
            在此版本的状态转移模型中, 每一步只对特定的目标作用【初始版本里面是随机选择一个目标作用】
        '''
        self.single_reward = 0
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # 0. 环境基本状态转移模型
        # 环境整体模型,对目标需求做更新
        self.integrate_state()
        # 1. 载荷及飞行成本模型 目前为每类飞机的成本固定为单位1
        self.calculate_cost()
        self.steps += 1

    def integrate_state(self):
        knowledge = Knowledge(num_requirement_type=self.num_requirement_type, num_plane_type=self.num_plane_type, num_target_type=self.num_target_type)
            
        # 加载智能体动作对基地的影响(对基地智能体中的实际飞机数目有影响)
        for i, agent in enumerate(self.agents):
            real_act = agent.action.act2real
            for j, plane_num in enumerate(real_act):
                if plane_num > 0:
                    real_index = knowledge.get_plane_type()[j]
                    agent.state.real_planes[real_index] -= plane_num 
        # 加载智能体动作对目标的影响
        for i, agent in enumerate(self.agents):
            resource_output = agent.action.act2source
            self.targets[self.target_focus_on].requirements -= resource_output
        # 判断是否解决当前任务
        self.focus_time += 1
        tar = self.targets[self.target_focus_on],
        if np.all(tar.state.requirements <= 0):
            self.is_focus_done = True
            self.targets_done[self.target_focus_on] = 1
            self.targets_type_remain[tar.type] -= 1 
            self.single_reward = 1
            print("目标{}的需求被解决".format(tar.index))
            print("该目标更新后的需求为{}".format(tar.state.requirements))

         # 对focus目标做更新 TODO 在世界定义的时候需要先更新focus
        if self.is_focus_done or self.focus_time >= self.num_plane_type - 1:
            idx = np.random.choice(len(self.target_index_list))
            self.target_focus_on = self.target_index_list[idx] # 随机选取一个target
            self.target_index_list = np.delete(self.target_index_list, idx)
            self.is_focus_done = False
            self.focus_time = 0
            self.single_cost = 0
    
    def calculate_cost(self):
        '''
            这个版本中认为每类飞机的成本都是固定的,都是c
        '''
        for i, agent in enumerate(self.agents):
            real_act = agent.action.act2real
            temp_cost = np.sum(real_act * ScenarioConfig.plane_cost)
            self.single_cost += temp_cost
            self.total_cost += temp_cost

    

    
