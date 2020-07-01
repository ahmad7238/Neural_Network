# Title: 1D Richards' equation
# Assumptions: Non-Homogeneous soil
#              Vertical soil block
#              Pressure head-based form
#              Uniform mesh size

import time
import matplotlib.pyplot as plt
import numpy as np

t0 = time.time()
lz = 1
lz_layer1 = 0.6
lz_layer2 = 0.3
lz_layer3 = 0.9
dt = 10
totaltime = 5 * 3600
elapsedtime = 0
max_err = 0.0001
maxiter = 200

# Mesh Generation
nzc = 100
nzv = nzc + 1
dz = lz / nzc

zv = np.zeros(nzv)
for i in list(range(nzv - 1)):
    zv[i + 1] = zv[i] + dz

zc = np.zeros(nzc)
for i in list(range(nzc)):
    zc[i] = 0.5 * (zv[i + 1] + zv[i])

# Code Acceleration Part
k_unsate = np.zeros(nzc)
k_unsatw = np.zeros(nzc)
ATDMA = np.zeros(nzc)
BTDMA = np.zeros(nzc)
CTDMA = np.zeros(nzc)
DTDMA = np.zeros(nzc)

theta_s = np.zeros(nzc)
theta_r = np.zeros(nzc)
alpha = np.zeros(nzc)
nv = np.zeros(nzc)
ks = np.zeros(nzc)
vm = np.zeros(nzc)

psi_n1 = np.zeros(nzc)
psi_n = np.zeros(nzc)
psi_n1_old = np.zeros(nzc)

# Mesh Generation (part 2)
for i in list(range(nzc)):
    if i < lz_layer1 * nzc:
        theta_s[i] = 0.366
        theta_r[i] = 0.029
        alpha[i] = 0.028 * 100
        nv[i] = 2.239
        ks[i] = 540.96 / 8640000
    elif (lz_layer1 * nzc) - 1 < i < (lz_layer3 * nzc) - 1:
        theta_s[i] = 0.467
        theta_r[i] = 0.106
        alpha[i] = 0.01 * 100
        nv[i] = 1.395
        ks[i] = 13.099 / 8640000
    else:
        theta_s[i] = 0.366
        theta_r[i] = 0.029
        alpha[i] = 0.028 * 100
        nv[i] = 2.239
        ks[i] = 540.96 / 8640000

# Initial Condition
psi_n1 = [-10 for i in list(range(nzc))]
psi_n = [-10 for i in list(range(nzc))]
psi_n1_old = [-10 for i in list(range(nzc))]


# Code Functions
def TDMAsolver(a, b, c, d):
    n = len(b)
    x = np.zeros(n)

    # Modify the first-row coefficient
    c[0] = c[0] / b[0]  # Division by zero risk
    d[0] = d[0] / b[0]  # Division by zero would imply a singular matrix
    for i in list(range(1, n - 1)):
        temp = b[i] - a[i] * c[i - 1]
        c[i] = c[i] / temp
        d[i] = (d[i] - a[i] * d[i - 1]) / temp

    d[n - 1] = (d[n - 1] - a[n - 1] * d[n - 2]) / (b[n - 1] - a[n - 1] * c[n - 2])
    x[n - 1] = d[n - 1]
    for i in list(range(n - 2, -1, -1)):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def VGSWC(psi, thetas, thetar, alfa, n_v):
    if psi < 0:
        m = 1 - 1 / n_v
        theta = ((thetas - thetar) / (1 + alfa * abs(psi) ** n_v) ** m) + thetar
    else:
        theta = thetas
    return theta


def k_unsat(psi, thetas, thetar, alfa, n_v, k_s):
    theta = VGSWC(psi, thetas, thetar, alfa, n_v)
    se = (theta - thetar) / (thetas - thetar)
    m = 1 - 1 / n_v
    kunsat = k_s * (se ** 0.5) * (1 - (1 - se ** (1 / m)) ** m) ** 2
    return kunsat


def Cw(psi, thetas, thetar, alfa, n_v):
    if psi < 0:
        cw = ((alfa ** n_v) * (thetas - thetar) * (n_v - 1) * ((-psi) ** (n_v - 1))) / (
                (1 + (alfa * (-psi)) ** n_v) ** (2 - (1 / n_v)))
    else:
        cw = 0
    return cw


# Governing Equation
while elapsedtime < totaltime:
    elapsedtime = elapsedtime + dt
    for niter in list(range(maxiter)):
        psi_err = 0

        # Bottom Boundary
        i = 0
        psi_bot_di = -10

        k_unsate[i] = 0.5 * (k_unsat(psi_n1[i + 1], theta_s[i], theta_r[i], alpha[i], nv[i], ks[i]) +
                             k_unsat(psi_n1[i], theta_s[i], theta_r[i], alpha[i], nv[i], ks[i]))
        k_unsatw[i] = 0.5 * (k_unsat(psi_bot_di, theta_s[i], theta_r[i], alpha[i], nv[i], ks[i])
                             + k_unsat(psi_n1[i], theta_s[i], theta_r[i], alpha[i], nv[i], ks[i]))

        ATDMA[i] = -k_unsatw[i] / dz
        BTDMA[i] = (dz * Cw(psi_n1[i], theta_s[i], theta_r[i], alpha[i], nv[i]) / dt) + k_unsate[i] / dz + \
                   2 * k_unsatw[i] / dz
        CTDMA[i] = -k_unsate[i] / dz
        DTDMA[i] = (dz * Cw(psi_n1[i], theta_s[i], theta_r[i], alpha[i], nv[i]) / dt) * psi_n[i] + \
                   k_unsate[i] - k_unsatw[i] + (k_unsatw[i] / (0.5 * dz)) * psi_bot_di

        for i in list(range(1, nzc - 1)):
            k_unsatw[i] = 0.5 * (k_unsat(psi_n1[i - 1], theta_s[i], theta_r[i], alpha[i], nv[i], ks[i]) +
                                 k_unsat(psi_n1[i], theta_s[i], theta_r[i], alpha[i], nv[i], ks[i]))
            k_unsate[i] = 0.5 * (k_unsat(psi_n1[i + 1], theta_s[i], theta_r[i], alpha[i], nv[i], ks[i])) \
                          + k_unsat(psi_n1[i], theta_s[i], theta_r[i], alpha[i], nv[i], ks[i])

            ATDMA[i] = -k_unsatw[i] / dz
            BTDMA[i] = (dz * Cw(psi_n1[i], theta_s[i], theta_r[i], alpha[i], nv[i]) / dt) + k_unsate[i] / dz + \
                       2 * k_unsatw[i] / dz
            CTDMA[i] = -k_unsate[i] / dz
            DTDMA[i] = (dz * Cw(psi_n1[i], theta_s[i], theta_r[i], alpha[i], nv[i]) / dt) * psi_n[i] \
                       + k_unsate[i] - k_unsatw[i] + (k_unsatw[i] / (0.5 * dz)) * psi_bot_di

        i = nzc - 1

        k_unsatw[i] = 0.5 * (k_unsat(psi_n1[i - 1], theta_s[i], theta_r[i], alpha[i], nv[i], ks[i]) +
                             k_unsat(psi_n1[i], theta_s[i], theta_r[i], alpha[i], nv[i], ks[i]))

        ATDMA[i] = -k_unsatw[i] / dz
        BTDMA[i] = (dz * Cw(psi_n1[i], theta_s[i], theta_r[i], alpha[i], nv[i]) / dt) + k_unsate[i] / dz \
                   + 2 * k_unsatw[i] / dz
        CTDMA[i] = 0
        DTDMA[i] = (dz * Cw(psi_n1[i], theta_s[i], theta_r[i], alpha[i], nv[i]) / dt) * psi_n[i] - \
                   k_unsatw[i] + (1.25 / 360000)

        psi_n1 = TDMAsolver(ATDMA, BTDMA, CTDMA, DTDMA)

        for i in list(range(nzc)):
            psi_err = abs(psi_n1_old[i] - psi_n1[i]) + psi_err
        if psi_err < max_err:
            break
        psi_n1_old = psi_n1

    if elapsedtime < totaltime:
        print('Elapsed Time: ', elapsedtime, ', iteration: ', niter)
        if elapsedtime + dt > totaltime:
            dt = totaltime - elapsedtime

    psi_n = psi_n1
    psi_n1_old = psi_n1

    vm = theta_r + ((theta_s - theta_r) / (1 + alpha * abs(psi_n1) ** nv) ** (1 - (1 / nv)))

fig = plt.figure()
plt.subplot(1, 2, 1)
plt.plot(psi_n1, zc)
plt.xlabel('Pressure Head (m)')
plt.ylabel('Depth (m)')

plt.subplot(1, 2, 2)
plt.plot(vm, zc)
plt.xlabel('Volumetric Moisture (m3/m3)')
plt.ylabel('Depth (m)')

# fig,(ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle()
# ax1.plot(psi_n1, zc)
# ax2.plot(vm, zc)

fig.savefig('Uni_project3.png')
plt.show()

t1 = time.time()
print('time required: ', (t1 - t0) / 60)
print()