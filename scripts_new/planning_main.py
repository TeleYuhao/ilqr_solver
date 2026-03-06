import numpy as np
from obstacle import obstacle
from KinematicModel import KinematicModel
# from Model_Parking import ModelParking
from Model_new import ModelParking
from ILQR_Core import ILQRCore
# vehicle parameter [m]
LENGTH = 4.5
WIDTH = 2.0
WB = 3.6
SAFETY_BUFFER = 1.5
STATE_DIM = 4
CONTROL_DIM = 2

config = {
    "v_max": 10.0,
    "v_min": 0.0,
    "acc_max": 2.0,
    "acc_min": -2.0,
    "delta_max": 1.57,
    "delta_min": -1.57,
    "horizon": 60,
    "dt": 0.1,
    "Q": np.diag([1.0,1.0,0.5,0]),
    "R": np.diag([1.0,1.0]),
    "ref_velo": 6.0,
    "state_dim": 4,
    "control_dim": 2,
    "wheelbase": WB,
    'max_iter': 50,
    'tol': 1e-3,
    'lamb_decay': 0.7,
    'lamb_amplify': 2.0,
    'max_lamb': 1e4,
    'alpha_options': [1., 0.5, 0.25, 0.125, 0.0625],
    'init_lamb': 20.0,
}

config_1 = {
    "v_max": 10.0,
    "v_min": 0.0,
    "acc_max": 2.0,
    "acc_min": -2.0,
    "omega_max": 1.57,
    "omega_min": -1.57,
    "horizon": 60,
    "dt": 0.1,
    "Q": np.diag([1.0,1.0,0.5,0, 100.0]),
    "R": np.diag([10.0,10.0]),
    "ref_velo": 6.0,
    "state_dim": 5,
    "control_dim": 2,
    "wheelbase": WB,
    'max_iter': 50,
    'tol': 1e-3,
    'lamb_decay': 0.7,
    'lamb_amplify': 2.0,
    'max_lamb': 1e4,
    'alpha_options': [1., 0.5, 0.25, 0.125, 0.0625],
    'init_lamb': 20.0,
}


if __name__== "__main__":
    ego_state = [0., 0., 5.0, 0.]   # [x y v yaw]
    
    longit_ref = np.linspace(0, 50, 1000)
    lateral_ref = np.linspace(0, 0, 1000)
    ref_waypoints = np.vstack((longit_ref, lateral_ref))
    ref_velo = np.array(6.0)

    obstacle_attr_1 = np.array([WIDTH, LENGTH, SAFETY_BUFFER]) # width length safe
    obstacle_attr_2 = np.array([WIDTH, LENGTH, SAFETY_BUFFER])
    obs_1 = [6.5, -0.2, 3.0, -0.0]
    obs_2 = [20, 4, 2.0, 0]

    obstacle_list = [obstacle(obs_1,obstacle_attr_1), obstacle(obs_2,obstacle_attr_2)]

    
    model = KinematicModel(config)
    init_control = np.zeros((60,2))
    init_x = model.init_traj(ego_state,init_control)
    print(model.compute_cost(init_x,init_control,ref_waypoints,obstacle_list))

    model.get_derivates(init_x,init_control,ref_waypoints,obstacle_list)
    Solver = ILQRCore(model,config)
    x_,u_ = Solver.solve(init_x,init_control,ref_waypoints,obstacle_list)
    

    ego_state = [0., 0., 5.0, 0.0 , 0.5]   # [x y v phi yaw]
    model_parking = ModelParking(config_1)
    init_control_parking = np.zeros((60,2))
    init_x_parking = model_parking.init_traj(ego_state,init_control_parking)
    Solver = ILQRCore(model_parking,config_1)
    x,u = Solver.solve(init_x_parking,init_control_parking,ref_waypoints,obstacle_list)
    import matplotlib.pyplot as plt
    plt.plot(x[:,0],x[:,1])
    plt.plot(x_[:,0],x_[:,1])
    plt.show()
