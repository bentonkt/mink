import numpy as np
import pyquaternion as pyq
from functools import partial
from scipy.spatial.transform import Rotation as R

class KeyCallback:
    """
    Class to handle keyboard input for teleoperation.
    The class provides methods to toggle teleoperation on/off,
    switch between manual and non-manual modes,
    adjust step sizes / speed for movement and rotation, 
    and select movements based on key presses.
    """
    
    def __init__(self, data):
        self.on = False
        self.data = data
        self.reset_state()
        self.actions = {
            78: self.toggle_manual,  # n: toggle non-manual mode
            46: self.toggle_rotation,  # .: toggle rotation mode
            56: self.toggle_mocap,  # m: toggle mocap data
            61: partial(self.toggle_speed, 1),  # +: increase speed
            45: partial(self.toggle_speed, -1),  # -: decrease speed
            265: partial(self.movement_select, 265, 0, 1),  # Up arrow
            264: partial(self.movement_select, 264, 0, -1),  # Down arrow
            262: partial(self.movement_select, 262, 1, 1),  # Right arrow
            263: partial(self.movement_select, 263, 1, -1),  # Left arrow
            55: partial(self.movement_select, 55, 2, 1),  # 6
            54: partial(self.movement_select, 54, 2, -1),  # 7
        }
        self.movements = {
            265: partial(self.movement_select, -1, 0, 1),  # Up arrow
            264: partial(self.movement_select, -1, 0, -1),  # Down arrow
            262: partial(self.movement_select, -1, 1, 1),  # Right arrow
            263: partial(self.movement_select, -1, 1, -1),  # Left arrow
            55: partial(self.movement_select, -1, 2, 1),  # 6
            54: partial(self.movement_select, -1, 2, -1),  # 7
        }
        self.opposite_keys = {
            265: 264,
            264: 265,
            262: 263,
            263: 262,
            55: 54,
            54: 55,
        }


    def key_callback_data(self, key):
        # Toggle teleop on/off
        if key == 57:  # 9
            self.toggle_on()
            return

        # Do nothing if teleop is off
        if not self.on:
            return

        if key in self.actions:
            self.actions[key]()


    def auto_key_move(self):
        """
        Automatically move the mocap body based on key presses.
        """

        if not self.on:
            return

        for key, action in self.movements.items():
            if self.keys[key]:
                action()


    def movement_select(self, key, axis, direction):
        """
        Select the movement direction based on the key pressed.
        """

        if not self.manual and key != -1:
            self.keys[key] = not self.keys[key]
            self.keys[self.opposite_keys[key]] = False
        elif not self.manual and key == -1:
            self.rot_or_trans(key, axis, direction)
        elif self.manual:
            self.rot_or_trans(key, axis, direction)


    def rot_or_trans(self, key, axis, direction):
        """
        Adjust the position or rotation of the mocap body based on rotation mode.
        """

        if self.rotation:
            self.adjust_rotation(key, axis, direction)
        else:
            self.adjust_position(key, axis, direction)


    def adjust_position(self, key, axis, direction):
        """
        Adjust the position of the mocap body in the specified direction
        based on the axis and step size.
        """

        # reorder from (w, x, y, z) to (x, y, z, w)
        q = self.data.mocap_quat[self.mocap_idx].copy()
        temp = q[0]
        q[:3] = q[1:]
        q[3] = temp

        rot = R.from_quat(q).as_matrix()
        unit_vec = rot[:, axis]
        step_size = self.m_step_size if self.manual else self.nm_step_size
        self.data.mocap_pos[self.mocap_idx] += direction * step_size * unit_vec


    def adjust_rotation(self, key, axis, direction):
        """
        Adjust the rotation of the mocap body in the specified direction
        based on the axis and step size.
        """

        step_size = self.m_rotation_step if self.manual else self.nm_rotation_step
        self.data.mocap_quat[self.mocap_idx] = self.rotate_quaternion(self.data.mocap_quat[self.mocap_idx], 
                                                         axis, 
                                                         direction * step_size)


    def rotate_quaternion(self, quat, axis, angle):
        """
        Rotate a quaternion by an angle around an axis.
        """

        unit_axis = [1, 0, 0]
        if axis == 1:
            unit_axis = [0, 1, 0]
        elif axis == 2:
            unit_axis = [0, 0, 1]

        angle_rad = np.deg2rad(angle)
        unit_axis = unit_axis / np.linalg.norm(unit_axis)
        q = pyq.Quaternion(quat)
        return (q * pyq.Quaternion(axis=unit_axis, angle=angle_rad)).elements
    
    
    def toggle_on(self):
        self.on = not self.on
        state = "On" if self.on else "Off"
        print(f"Keyboard Teleoperation toggled: {state}!")
        self.reset_state()
        print()


    def toggle_manual(self):
        self.manual = not self.manual
        manual_state = "On" if self.manual else "Off"
        print(f"Manual mode toggled: {manual_state}!")
        self.reset_keys()
        print()


    def toggle_rotation(self):
        self.rotation = not self.rotation
        state = "On" if self.rotation else "Off"
        print(f"Rotation mode toggled: {state}!")
        self.reset_keys()
        print()


    def toggle_speed(self, direction):
        factor = 1.10 if direction == 1 else 0.9
        if self.manual:
            if self.rotation:
                self.m_rotation_step *= factor
            else:
                self.m_step_size *= factor
        else:
            if self.rotation:
                self.nm_rotation_step *= factor
            else:
                self.nm_step_size *= factor
        
        output = "Manual" if self.manual else "Non-manual"
        mode = "Rotation" if self.rotation else "Translation"
        if self.manual:
            step_size = self.m_rotation_step if self.rotation else self.m_step_size
        else:
            step_size = self.nm_rotation_step if self.rotation else self.nm_step_size
        print(f"{output} {mode} step size: {step_size:.8f}")

    
    def toggle_mocap(self):
        self.mocap_idx = (self.mocap_idx + 1) % self.data.mocap_pos.shape[0] # cycle through mocap data
        print(f"Current mocap index: {self.mocap_idx}")


    def reset_keys(self):
        self.keys = {
            265: False,
            264: False,
            262: False,
            263: False,
            55: False,
            54: False,
        }


    def reset_step_size(self):
        self.m_step_size = 0.01  # manual step size
        self.m_rotation_step = 10  # manual rotation step
        self.nm_step_size = 1e-4  # non-manual step size
        self.nm_rotation_step = 5e-2  # non-manual rotation step
        print("Step sizes have been reset!")


    def reset_state(self):
        self.reset_keys()
        self.reset_step_size()
        self.manual = True
        self.rotation = False
        self.mocap_idx = 0
        str = f"States have been reset: \n \
        - Manual mode: {self.manual} \
        - Rotation mode: {self.rotation} \n \
        - Mocap index: {self.mocap_idx}"

        print(str)