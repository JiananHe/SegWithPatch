import numpy as np
import matplotlib.pyplot as plt


def make_curve():
    x = np.arange(1, 1001)
    y1 = np.log(x[:60]) / np.log(0.0010874632336580173) + 1.2
    y1 += 0.05 * np.random.normal(0, 1, 60)
    y2 = (-0.5 / 940) * x[60:] + (0.6 + 0.5 * 6/94)
    y2 += 0.01 * np.random.normal(0, 1, 940)

    y = np.concatenate((y1, y2))
    return x, y


def exp_wgt_avg(y, b):
    v = np.mean(y[:5])
    vx = [0]
    vy = [v]
    for i, n in enumerate(y[5:]):
        v = (b * v + (1 - b) * n)
        vx.append(i+1)
        vy.append(v)
        # print(v)

    return np.array(vx), np.array(vy)


if __name__ == '__main__':
    x1, y1 = make_curve()
    b1 = 0.8
    vx1, vy1 = exp_wgt_avg(y1, b1)

    plt.figure()
    # plt.plot(x1, y1)
    # plt.plot(x, y, 'o')
    plt.plot(vx1, vy1, 'r')
    # plt.plot(vx2, vy2, 'b')
    plt.show()