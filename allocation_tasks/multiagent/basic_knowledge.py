import numpy as np

class ScenarioConfig(object):  
    '''
        场景参数放置于此处
    '''
    
    num_agents = 5
    max_episode_step = 500
    num_requirement_type = 7
    num_plane_type = 12
    max_num_plane = 4
    num_target_type = 6

    plane_cost = np.ones(num_plane_type) * 0.2 # 每个飞机的成本，暂定
    single_target_reward = 1  # 完成每个target后给的奖励，待定
    total_reward = 500 # episode结束后得到的大的稀疏奖励，待定

class Knowledge():
    def __init__(self, num_requirement_type, num_plane_type, num_target_type):
        '''
            此处代码用来存储场景中的已知信息
        '''
        self.num_requirement_type = num_requirement_type # 有几类需求飞机能力向量和目标需求向量的维度就是几维
        self.num_plane_type = num_plane_type # 这里是飞机挂载的类型而非飞机类型
        self.num_target_type = num_target_type
        self.plane_type = {}

        # A. 存储每类飞机挂载选择的能力向量
        self.PLANE_CAPACITY = np.array([
            [0,4,0,0,0,0,0], # e.g.第一类飞机挂载种类
            [4,0,0,0,0,0,0], 
            [0,0,2,0,0,0,0], 
            [0,0,0,0,2,0,0], 
            [0,0,0,4,0,0,0],
            [0,0,0,0,0,4,0],
            [0,0,0,0,2,0,0],
            [0,0,0,2,0,0,0],
            [0,0,0,0,0,6,0],
            [0,0,0,0,0,0,4],
            [0,0,0,4,0,0,0],
            [0,0,0,0,0,0,2],
        ])
        assert  self.PLANE_CAPACITY.shape[-1] == self.num_requirement_type, ("PLANE_CAPACITY 的格式有问题！")
        assert  self.PLANE_CAPACITY.shape[-2] == self.num_plane_type, ("PLANE_CAPACITY 的格式有问题！")

        # B. 存储每类飞机挂载实际对应的飞机类型
        self.REAL_PLANE = np.array([
            0,  # 飞机类型1
            0,
            0,
            1,  # 飞机类型2 
            1,
            1,
            2,  # 飞机类型3 
            2,
            2,
            2,
            3,  # 飞机类型4
            4,  # 飞机类型5
        ])
        assert self.REAL_PLANE.shape[-1] == num_plane_type, ("REAL_PLANE格式有问题！")

        # C. 存储每类子目标的需求向量
        self.TARGET_REQUIREMENT = np.array([
            [4,4,0,0,0,0,0], # e.g. 第一类目标需求种类
            [0,0,2,0,0,0,0], 
            [0,0,0,2,0,0,2], 
            [0,0,0,4,6,0,0], 
            [0,0,0,0,0,6,2], 
            [0,0,0,4,0,4,0], 

        ])
        assert self.TARGET_REQUIREMENT.shape[-1] == self.num_requirement_type, ("TARGET_REQUIREMENT 的格式有问题！")
        assert self.TARGET_REQUIREMENT.shape[-2] == self.num_target_type, ("TARGET_REQUIREMENT 的格式有问题！")

        # D. 存储每个基地智能体的飞机种类（此处为实际挂载种类）
        self.AGENT_PLANE = np.array([
            [36,36,36,36,36,36,0,0,0,0,0,0],  # e.g. 机场1中有实际飞机类型1和2，对应了6种挂载类型      
            [0,0,0,36,36,36,0,0,0,0,0,24],    
            [0,0,0,36,36,36,48,48,48,48,0,36],
            [24,24,24,36,36,36,48,48,48,48,24,0],
            [0,0,0,0,0,0,0,0,0,0,24,24], 

        ])
        assert self.AGENT_PLANE.shape[-1] == self.num_plane_type, ("AGENT_PLANE 的格式有问题！")
        
        # E. 存储每个基地智能体可调用的飞机阈值（此处为真飞机种类）
        self.AGENT_THRES = np.array([
            [20,20,0,0,0],  # e.g. 机场1实际有[36,36,0,0,0]的真飞机，能调用的上限为[20,20,0,0,0]  
            [0,20,0,0,18],  
            [0,20,36,0,20],
            [18,20,36,18,0],
            [0,0,0,18,18],  
        ])
        assert self.AGENT_THRES.shape[-1] > 0 , ("AGENT_THRES 的格式有问题！")

    def get_plane_capacity(self):
        '''
            返回每类飞机挂载选择的能力向量
        '''
        return self.PLANE_CAPACITY
    def get_plane_type(self):
        ''' 
            返回每类飞机挂载实际对应的飞机类型
        '''
        return self.REAL_PLANE
    def get_target_requirement(self):
        '''
            返回每类子目标的需求向量
        '''
        return self.TARGET_REQUIREMENT
    def get_agent_plane(self):
        '''
            存储每个基地智能体的飞机种类
        '''
        return self.AGENT_PLANE
    def get_agent_thres(self):
        '''
            存储每个基地智能体可调用的飞机阈值
        '''
        return self.AGENT_THRES



if __name__ == '__main__':
    TEMP = Knowledge(num_requirement_type=7, num_plane_type=12, num_target_type=6)
    test = TEMP.get_plane_capacity()
    test = TEMP.get_plane_type()
    test = TEMP.get_target_requirement()
    test = TEMP.get_agent_plane()
    test = TEMP.get_agent_thres()

 