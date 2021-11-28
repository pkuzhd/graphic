import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# X, Y value
X = np.arange(-4, 4.25, 0.25)
Y = np.arange(-4, 4.25, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.abs(X) + np.abs(Y)
X_p = np.abs(X)
Y_p = np.abs(Y)
# height value
Z = X_p
for i in range(len(Z)):
    for j in range(len(Z[i])):
        Z[i][j] = max(Z[i][j], Y_p[i][j])
Z = -Z
# rstride:行之间的跨度  cstride:列之间的跨度
# rcount:设置间隔个数，默认50个，ccount:列的间隔个数  不能与上面两个参数同时出现
# vmax和vmin  颜色的最大值和最小值
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap=plt.get_cmap('rainbow')
                )
# zdir : 'z' | 'x' | 'y' 表示把等高线图投射到哪个面
# offset : 表示等高线图投射到指定页面的某个刻度
# ax.contourf(X, Y, Z, zdir='z', offset=-2)
# 设置图像z轴的显示范围，x、y轴设置方式相同
# ax.set_zlim(-2, 2)

t = np.linspace(0, np.pi * 2, 20)
s = np.linspace(0, np.pi, 20)

r = 0.3
t, s = np.meshgrid(t, s)
x = np.cos(t) * np.sin(s) * r
y = np.sin(t) * np.sin(s) * r
z = np.cos(s) * r

cameras = [(-2, 0, 4), (2, 0, 4), (0, 0, 6)]
colors = [(255 / 256, 168 / 256, 168 / 256), (168 / 256, 191 / 256, 255 / 256), (176 / 256, 214 / 256, 131 / 256)]
for idx, camera in enumerate(cameras):
    ax.plot_surface(x + camera[0], y + camera[1], z + camera[2], rstride=1, cstride=1, color=colors[idx])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='rectilinear')
# ax.plot([1, 2, 3, 4])
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
ax.set_xlabel('x axis')  # x轴名称
ax.set_ylabel('y axis')  # y轴名称
ax.set_zlabel('z axis')
ax.view_init(135, -90)


def get_x_y_d(X, Y, Z, i, j, pos):
    x = X[i][j] - pos[0]
    y = Y[i][j] - pos[1]
    z = Z[i][j] - pos[2]
    u = -math.atan(x / z)
    v = math.atan(y / math.sqrt(z ** 2 + x ** 2))
    return u, v, math.sqrt(x ** 2 + y ** 2 + z ** 2)


def get_x_y_d_2(x, y, z):
    u = -math.atan(x / z)
    v = math.atan(y / math.sqrt(z ** 2 + x ** 2))
    return u, v


def place_x_y(uu, vv, d, origin_pos, new_pos):
    if d == np.inf:
        u = uu
        v = vv
    else:
        x = d * math.cos(vv) * math.sin(uu) + origin_pos[0]
        z = d * math.cos(vv) * math.cos(uu) + origin_pos[2]
        y = d * math.sin(vv) + origin_pos[1]
        u, v = get_x_y_d_2(x, y, z)
    return u, v

# print(place_x_y(math.pi, 0, 10, cameras[0], cameras[2]))
# exit(0)
X_cam = [np.array(X), np.array(X), np.array(X)]
Y_cam = [np.array(Y), np.array(Y), np.array(Y)]
D_cam = [np.array(Y), np.array(Y), np.array(Y)]

for idx, camera in enumerate(cameras):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_cam[idx][i][j], Y_cam[idx][i][j], D_cam[idx][i][j] = get_x_y_d(X, Y, Z, i, j, camera)

for idx, camera in enumerate(cameras):
    fig = plt.figure()

    for i in range(X.shape[0] - 1):
        for j in range(X.shape[1]):
            x = [0, 0]
            y = [0, 0]
            x[0], y[0] = X_cam[idx][i][j], Y_cam[idx][i][j]
            x[1], y[1] = X_cam[idx][i + 1][j], Y_cam[idx][i + 1][j]
            plt.plot(x, y, color=colors[idx])
    for i in range(X.shape[0]):
        for j in range(X.shape[1] - 1):
            x = [0, 0]
            y = [0, 0]
            x[0], y[0] = X_cam[idx][i][j], Y_cam[idx][i][j]
            x[1], y[1] = X_cam[idx][i][j + 1], Y_cam[idx][i][j + 1]
            plt.plot(x, y, color=colors[idx])
    # plt.plot([-math.pi, math.pi, math.pi, -math.pi, -math.pi],
    #          [-math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2], color='black')
    plt.title(["真实A", "真实B", "真实C"][idx])
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.5, 0.5)

d = 10
for idx, camera in enumerate(cameras[:2]):
    fig = plt.figure()

    for i in range(X.shape[0] - 1):
        for j in range(X.shape[1]):
            x = [0, 0]
            y = [0, 0]
            x[0], y[0] = place_x_y(X_cam[idx][i][j], Y_cam[idx][i][j], d, camera, cameras[2])
            x[1], y[1] = place_x_y(X_cam[idx][i + 1][j], Y_cam[idx][i + 1][j], d, camera, cameras[2])
            plt.plot(x, y, color=colors[idx])
    for i in range(X.shape[0]):
        for j in range(X.shape[1] - 1):
            x = [0, 0]
            y = [0, 0]
            x[0], y[0] = place_x_y(X_cam[idx][i][j], Y_cam[idx][i][j], d, camera, cameras[2])
            x[1], y[1] = place_x_y(X_cam[idx][i][j + 1], Y_cam[idx][i][j + 1], d, camera, cameras[2])
            plt.plot(x, y, color=colors[idx])
    # plt.plot([-math.pi, math.pi, math.pi, -math.pi, -math.pi],
    #          [-math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2], color='black')
    plt.title(["虚拟A d=无穷", "虚拟B d=无穷"][idx])
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.5, 0.5)

fig = plt.figure()
for idx, camera in enumerate(cameras[:2]):
    for i in range(X.shape[0] - 1):
        for j in range(X.shape[1]):
            x = [0, 0]
            y = [0, 0]
            x[0], y[0] = place_x_y(X_cam[idx][i][j], Y_cam[idx][i][j], d, camera, cameras[2])
            x[1], y[1] = place_x_y(X_cam[idx][i + 1][j], Y_cam[idx][i + 1][j], d, camera, cameras[2])
            plt.plot(x, y, color=colors[idx])
    for i in range(X.shape[0]):
        for j in range(X.shape[1] - 1):
            x = [0, 0]
            y = [0, 0]
            x[0], y[0] = place_x_y(X_cam[idx][i][j], Y_cam[idx][i][j], d, camera, cameras[2])
            x[1], y[1] = place_x_y(X_cam[idx][i][j + 1], Y_cam[idx][i][j + 1], d, camera, cameras[2])
            plt.plot(x, y, color=colors[idx], alpha=0.5)
    # plt.plot([-math.pi, math.pi, math.pi, -math.pi, -math.pi],
    #          [-math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2], color='black')
plt.title("虚拟AB d=无穷")
plt.xlim(-0.7, 0.7)
plt.ylim(-0.5, 0.5)

fig = plt.figure()
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x1, y1 = place_x_y(X_cam[0][i][j], Y_cam[0][i][j], d, camera, cameras[2])
        x2, y2 = place_x_y(X_cam[1][i][j], Y_cam[1][i][j], d, camera, cameras[2])
        plt.scatter(D_cam[2][i][j], x1 - x2, color='black', s=0.5)
plt.title("C相机深度与u偏差的关系")
plt.xlabel('深度')
plt.ylabel('u偏差')

fig = plt.figure()
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x1, y1 = place_x_y(X_cam[0][i][j], Y_cam[0][i][j], d, camera, cameras[2])
        x2, y2 = place_x_y(X_cam[1][i][j], Y_cam[1][i][j], d, camera, cameras[2])
        plt.scatter(D_cam[2][i][j], y1 - y2, color='black', s=0.5)
plt.title("C相机深度与v偏差的关系")
plt.xlabel('深度')
plt.ylabel('v偏差')

plt.show()
