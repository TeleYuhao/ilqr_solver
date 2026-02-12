from state_cost import StateCost
from ControlConstraint import ControlConstraint
from StateConstraint import StateConstraint
from obstacle_base import obstacle
from kinematic_mode import KinematicModel
from CostCalculator import CostCalculator
from ILQR_Core import ilqr
import numpy as np
DT = 0.1
HORIZON_LENGTH = 60
MAX_ACC = 2.0

# vehicle parameter [m]
LENGTH = 4.5
WIDTH = 2.0
WB = 3.6
SAFETY_BUFFER = 1.5
STATE_DIM = 4
CONTROL_DIM = 2

Q = np.diag([1.0,1.0,0.5,0])
R = np.diag([1.0,1.0])

if __name__ == "__main__":
    ego_state = [0., 0., 5.0, 0.]   # [x y v yaw]
    
    longit_ref = np.linspace(0, 50, 1000)
    lateral_ref = np.linspace(0, 0, 1000)
    ref_waypoints = np.vstack((longit_ref, lateral_ref))
    ref_velo = np.array(6.0)

    obstacle_attr_1 = np.array([WIDTH, LENGTH, SAFETY_BUFFER]) # width length safe
    obstacle_attr_2 = np.array([WIDTH, LENGTH, SAFETY_BUFFER])
    obs_1 = [6.5, -0.2, 3.0, 0.]
    obs_2 = [20, 4, 2.0, 0.]

    obstacle_list = [obstacle(obs_1,obstacle_attr_1), obstacle(obs_2,obstacle_attr_2)]
    state_cost = StateCost(Q,R, ref_waypoints,STATE_DIM,CONTROL_DIM)
    control_constraint = ControlConstraint(STATE_DIM,CONTROL_DIM)
    vehicle = KinematicModel() 
    state_constraint = StateConstraint(STATE_DIM,CONTROL_DIM,vehicle,obstacle_list)

    init_control = np.zeros((60,2))
    init_x = vehicle.init_traj(ego_state,init_control)

    cost_calculator = CostCalculator(state_cost,
                                    state_constraint,
                                    control_constraint,
                                    HORIZON_LENGTH,
                                    STATE_DIM,
                                    CONTROL_DIM)
    cost_calculator.StateCost.get_ref_states(init_x[:,:2])

    # cost = cost_calculator.CalculateTotalCost(init_x,init_control)
    # cost_calculator.CalculateDerivates(init_x,init_control)

    ilqr_solver = ilqr(vehicle,cost_calculator)
    init_J = cost_calculator.CalculateTotalCost(init_x,init_control)
    # ilqr_solver.backwardpass(init_control,init_x,lamb=20)
    # ilqr_solver.iter(init_control,init_x,init_J,lamb=20)
    u,x = ilqr_solver.solve(ego_state)
    import matplotlib.pyplot  as plt
    plt.plot(x[:,0],x[:,1])
    plt.show()
