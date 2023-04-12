#!/usr/bin/env python3

# PythonRobotic/PathTracking/pure_pursuit/pure_pursuit_diff.py
# 参考 PythonRobotic/PathTracking/pure_pursuit/pure_pursuit.py

import numpy as np
import math
import matplotlib.pyplot as plt


# Parameters
k = 0.1  # look forward gain
Lfc = 2.0  # [m] look-ahead distance
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time tick
r_min = 2 # 最小转弯半径
# 向心加速度英文: centripetal acceleration
ca_max = 0.5 ** 2 / r_min # 最大向心加速度 

look_ahead_min_distance = 2
look_ahead_max_distance = 4
look_ahead_distance_ratio = 2
frequency = 10

wheel_base = 0.5
show_animation = True

dt = 1 / frequency

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, angular=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.angular = angular

    def update(self, a, cv):
        dyaw = self.angular * dt
        self.x += self.v * math.cos(self.yaw + dyaw / 2) * dt
        self.y += self.v * math.sin(self.yaw + dyaw / 2) * dt
        self.yaw += dyaw
        self.v += a * dt
        self.angular = self.v * cv
            
        # 根据曲率和线速度计算角速度
        # TODO 约束线速度

    def calc_distance(self, point_x, point_y):
        dx = self.x - point_x
        dy = self.y - point_y
        return math.hypot(dx, dy)

class States:

    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.angular = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.angular.append(state.angular)
        self.t.append(t)

class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.last_point_index = None

    def search_target_index(self, state):
        if self.last_point_index is None:
            # 寻找最近的点
            dx = [state.x - icx for icx in self.cx]
            dy = [state.y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.last_point_index = ind
        else:
            ind = self.last_point_index
            distance = state.calc_distance(self.cx[ind], self.cy[ind])
            while True:
                distance_next = state.calc_distance(self.cx[ind + 1], self.cy[ind + 1])
                if distance < distance_next:
                    break
                if ind + 1 < len(self.cx):
                    ind = ind + 1
                distance = distance_next
            self.last_point_index = ind
        
        Lf = calculate_look_ahead_distance(state.v)

        # 取预瞄点
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if ind + 1 >= len(self.cx):
                break
            ind += 1
        
        return ind, Lf



def proportional_control(target, current):
    a = Kp * (target - current)

    return a

def pure_pursuit_diff_control(state, trajectory, pind):
    ind, Lf = trajectory.search_target_index(state)

    # 不往后
    if pind >= ind:
        ind = pind

    # 目标位置
    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:  # 到目标点
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    # 目标连线和车体夹角
    p0 = [state.x, state.y]
    p0_yaw = state.yaw
    p1 = [tx, ty]

    # 用公式把p1转换到p0方向为x轴的坐标系下
    p1_x = math.cos(p0_yaw) * (p1[0] - p0[0]) + math.sin(p0_yaw) * (p1[1] - p0[1])
    p1_y = -math.sin(p0_yaw) * (p1[0] - p0[0]) + math.cos(p0_yaw) * (p1[1] - p0[1])


    cv =  2 * (p1_y) / (p1_x ** 2 + p1_y ** 2)

    return cv, ind


# 限制值在min和max之间
def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def calculate_look_ahead_distance(v):
    return clamp(v * look_ahead_distance_ratio, look_ahead_min_distance, look_ahead_max_distance)


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def main():
#  target course
    cx = np.arange(0, 50, 0.5)
    cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]

    target_speed = 10.0 / 3.6  # [m/s]

    T = 100.0  # max simulation time

    # initial state
    state = State(x=-0.0, y=-3.0, yaw=0.0, v=0.0)

    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state)
    target_course = TargetCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(state)

    while T >= time and lastIndex > target_ind:

        # Calc control input
        ai = proportional_control(target_speed, state.v)
        di, target_ind = pure_pursuit_diff_control(
            state, target_course, target_ind)

        state.update(ai, di)  # Control vehicle

        time += dt
        states.append(time, state)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(states.x, states.y, "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[m/s]:" + str(state.v)[:4])
            plt.pause(1/frequency/10)

    # Test
    assert lastIndex >= target_ind, "Cannot goal"

    if show_animation:  # pragma: no cover
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(states.x, states.y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(states.t, [iv for iv in states.v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[m/s]")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(states.t, [ag for ag in states.angular], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Angular[rad/s]")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()