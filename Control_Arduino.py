from DeltaArray import DeltaArray
import numpy as np
import time

class Control_Delta():
    def __init__(self):
        self.delta = DeltaArray('/dev/tty.usbmodem141201')

    def goto_pos(self, pt, delta_number=4): # delta_number = 4 for actuators 10-12
        # pt: numpy array of 3 values
        position = [0.0] * 12
        offset = (delta_number - 1) * 3
        for i in range(3):
            position[i + offset] = pt[i] / 100 # convert from cm to m
        duration = [1.0]
        self.delta.move_joint_position(np.array([position]), duration)
        self.delta.wait_until_done_moving()
        positions = [round(100*float(p), 2) for p in self.delta.get_joint_positions()[9:]]
        return positions
        # print(self.delta.get_joint_positions())