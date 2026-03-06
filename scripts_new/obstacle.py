import numpy as np


class obstacle:
    def __init__(self,state,attr):
        self.state = state
        self.attr = attr
        self.pos = np.array([state[0],state[1]])
        self.yaw = state[3]
        self.ego_width = 2.0
        self.ego_pnt_radius = self.ego_width/2

        self.obs_width = attr[0]
        self.obs_length = attr[1]
        self.d_safe = attr[2]
        self.get_ellipsoid_obstacle_scales()
        self.prediction_traj = self.const_velo_prediction(self.state, 60)

    def get_ellipsoid_obstacle_scales(self):
        self.a = 0.5 * self.obs_length + self.d_safe + self.ego_pnt_radius
        self.b = 0.5 * self.obs_width + self.d_safe + self.ego_pnt_radius
    def ellipsoid_safety_margin(self,pnt, elp_center):
        theta = self.yaw

        diff = pnt - elp_center
        rotation_matrx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pnt_std = diff @ rotation_matrx  # rotate by (-theta)

        result = 1 - ((pnt_std[0] ** 2) / (self.a ** 2) + (pnt_std[1] ** 2) / (self.b ** 2))

        return result
    
    def ellipsoid_safety_margin_derivatives(self,pnt, elp_center):
        theta = self.yaw
        diff = pnt - elp_center
        rotation_matrx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pnt_std = diff @ rotation_matrx  # rotate by (-theta)

        # (1) constraint over standard point vec.:
        #       [c -> x_std, c -> y_std]
        res_over_pnt_std = np.array([-2 * pnt_std[0] / (self.a ** 2), -2 * pnt_std[1] / (self.b ** 2)])

        # (2) standard point vec. over difference vec.:
        #       [[x_std -> x_diff, x_std -> y_diff]
        #        [y_std -> x_diff, y_std -> y_diff]]
        pnt_std_over_diff = rotation_matrx.transpose()

        # (3) difference vec. over original point vec.:
        #       [[x_diff -> x, x_diff -> y]
        #        [y_diff -> x, y_diff -> y]]
        diff_over_pnt = np.eye(2)

        # chain (1)(2)(3) together:
        #       [c -> x, c -> y]
        res_over_pnt = res_over_pnt_std @ pnt_std_over_diff @ diff_over_pnt

        return res_over_pnt
    @staticmethod
    def kinematic_propagate(x0) -> np.array:
        x,y,v,yaw = x0
        return np.array([x + v * np.cos(yaw) * 0.1, y + v * np.sin(yaw) * 0.1, v, yaw])
    def const_velo_prediction(self,x0, steps: int) -> np.matrix:
        predicted_states = [x0]
        cur_x = x0
        for i in range(steps):
            next_x = self.kinematic_propagate(cur_x)
            cur_x = next_x
            predicted_states.append(next_x)

        predicted_states = np.vstack(predicted_states)

        return predicted_states
