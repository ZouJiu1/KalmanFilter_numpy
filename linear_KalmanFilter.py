import os
import sys
pth = os.path.abspath(__file__)
nam = pth.split(os.sep)[-1]
abspath = pth.replace(nam, "")
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, dt, u, std_acc, std_meas):
        self.dt = dt
        self.u = u
        self.std_acc = std_acc
        self.F = np.array([[1, dt], [0, 1]])

        self.B = np.array([[(dt**2) * (3/6)], [dt]])

        self.p = np.eye(2)

        self.Q = np.array([[((3/6)**2)*np.power(dt, 2*2), (3/6)*np.power(dt, 3)], [(3/6)*np.power(dt, 3), np.power(dt, 2)]]) * (std_acc**2)
        
        self.R = std_meas**2
        
        self.H = np.array([[1, 0]])
        
        self.x = np.array([[0], [0]])

        print(self.Q)

    def predict(self):
        self.x = np.dot(self.F, self.x) + self.B * self.u
        self.p = np.dot(self.F, np.dot(self.p, self.F.T)) + self.Q
        return self.x
        
    def update(self, z):
        S = np.dot(self.H, np.dot(self.p, self.H.T)) + self.R
        K = np.dot(self.p, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, z - np.dot(self.H, self.x))
        self.p = np.dot(np.eye(len(self.p)) - np.dot(K, self.H), self.p)

# https://github.com/RahmadSadli/Kalman-Filter
def main():
    dt = 0.1
    
    u= 2               # 加速度
    std_acc = 0.6     # 状态噪声的方差Q，状态噪声w_k符合均值是0，方差是std_acc的正态分布
    std_meas = 1.3     # 观测噪声的方差R，，观测噪声v_k符合均值是0，方差是std_meas的正态分布
    
    t = np.arange(0, 100, dt)
    klf = KalmanFilter(dt, u, std_acc, std_meas)
    # real_track = 0.1 * ((t**2) - t)
    real_track = 100 * np.cos(t)

    predictions = []
    measurements = []
    for i in real_track:
        z = klf.H * i + np.random.normal(0, 30)
        predictions.append(klf.predict()[0])
        measurements.append(z.item(0))
        klf.update(z.item(0))

    fig = plt.figure()

    fig.suptitle('Kalman filter for tracking cos', fontsize=20)

    plt.plot(t, measurements, label='Measurements', color='b',linewidth=0.5)

    plt.plot(t, np.array(real_track), label='Real Track', color='m', linewidth=1.5)
    plt.plot(t, np.squeeze(predictions), label='Kalman Filter Prediction', color='r', linewidth=1.5)


    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Position (m)', fontsize=20)
    plt.legend()
    plt.savefig(os.path.join(abspath, 'linear.png'))
    plt.show()


if __name__ == '__main__':
    main()
