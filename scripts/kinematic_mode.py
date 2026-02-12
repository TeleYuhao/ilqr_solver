from model_base import Model
import numpy as np
class KinematicModel(Model):
    def __init__(self):
        super().__init__(4,2)
        self.WheelBase = 3.6
        self.Width = 2.0
        self.Length = 4.5

    def gradient_fu(self, state, control, step=0.1):
        delta = control[1]
        yaw = state[3]
        v = state[2]
        beta_d = np.arctan(np.tan(delta / 2))
        beta_over_stl = 0.5 * (1 + np.tan(delta)**2) / (1 + 0.25 * np.tan(delta)**2)
        dfdu = np.array([
            [0,v * (-np.sin(beta_d + yaw)) * step * beta_over_stl],
            [0,v * np.cos(beta_d + yaw) * step * beta_over_stl],
            [step,0],
            [0,(2 * v * step / self.WheelBase) * np.cos(beta_d) * beta_over_stl]
        ])
        return dfdu
    def gradient_fx(self, state, control, step = 0.1):
        yaw = state[3]
        v = state[2]
        delta = control[1]
        beta_d = np.arctan(np.tan(delta / 2))
        dfdx = np.array([
            [1,0,np.cos(yaw + beta_d) * step, -v * np.sin(yaw + beta_d) * step],
            [0,1,np.sin(yaw + beta_d) * step, v * np.cos(yaw + beta_d) * step],
            [0,0,1,0],
            [0,0,2 * np.sin(beta_d) * step/self.WheelBase ,1]
        ])
        return dfdx
    def forward_calculation(self, state, control, step=0.1):
        beta = np.arctan(np.tan(control[1]) / 2)
        next_x = np.array([
            state[0] + state[2] * np.cos(beta + state[3]) * step,
            state[1] + state[2] * np.sin(beta + state[3]) * step,
            state[2] + control[0] * step,
            state[3] + 2 * state[2] * np.sin(beta) * step / self.WheelBase
        ])
        return next_x
    
    def get_vehicle_front_and_rear_center_derivatives(self,yaw):
        half_whba = 0.5 * self.WheelBase

        front_pnt_over_state = np.array([
            [1., 0., 0., half_whba * (-np.sin(yaw))],
            [0., 1., 0., half_whba *   np.cos(yaw) ]
        ])

        # rear point over (center) state:
        #            <similarly...>
        rear_point_over_state = np.array([
            [1., 0., 0., -half_whba * (-np.sin(yaw))],
            [0., 1., 0., -half_whba *   np.cos(yaw) ]
        ])

        return front_pnt_over_state, rear_point_over_state
    def get_vehicle_front_and_rear_centers(self,pos, yaw):
        half_whba_vec = 0.5 * self.WheelBase * np.array([np.cos(yaw), np.sin(yaw)])
        front_pnt = pos + half_whba_vec
        rear_pnt = pos - half_whba_vec

        return front_pnt, rear_pnt
    
    def init_traj(self, init_state, controls,horizon=60):
        states = np.zeros((horizon + 1, 4))
        states[0] = init_state
        for i in range(1,horizon+1):
            states[i] = self.forward_calculation(states[i-1], controls[i-1])
        return states

    
if __name__ == "__main__":
    model = KinematicModel()
    
