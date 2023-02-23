import numpy as np
class Knowledge():
    def __init__(self, num_requirement_type, num_plane_type):
        '''
            此处代码用来存储场景中的已知信息
            TO 李东珉, 需要运行这个里面的main以检测各知识的构造是否正确
        '''
        self.num_requirement_type = num_requirement_type # 有几类需求飞机能力向量和目标需求向量的维度就是几维
        self.num_plane_type = num_plane_type # 这里是飞机挂载的类型而非飞机类型
        self.plane_type = {}

        # A. 存储每类飞机挂载选择的能力向量
        self.PLANE_CAPACITY = np.array([
            [0,4,0,0,0,0,0], # e.g.第一类飞机挂载种类







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






        ])
        assert self.TARGET_REQUIREMENT.shape[-1] == self.num_requirement_type, ("TARGET_REQUIREMENT 的格式有问题！")

        # D. 存储每个基地智能体的飞机种类（此处为实际挂载种类）
        self.AGENT_PLANE = np.array([
            [36,36,36,36,36,36,0,0,0,0,0,0],  # e.g. 机场1中有实际飞机类型1和2，对应了6种挂载类型      




        ])
        assert self.AGENT_PLANE.shape[-1] == self.num_requirement_type, ("AGENT_PLANE 的格式有问题！")
        
    

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



if __name__ == '__main__':
    TEMP = Knowledge(num_requirement_type=7, num_plane_type=12)
    test = TEMP.get_plane_capacity()
    test = TEMP.get_plane_type()
    test = TEMP.get_target_capacity()
    test = TEMP.get_agent_plane()

 