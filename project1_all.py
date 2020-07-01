
# # Title: Class Project(Problem1)
#   Assumptions: Prismatic channel
#                fix lateral flow
#                Rough bed (By only adding tolh to initial condition)
#                Trapezoidal cross-section

# settings
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import xlwt
import time

t0 = time.time()

S0 = 0.0005
Tl = 200 * 60
HnL = 0.33 / 60000
sideslope = 0
n = 0.03
L = 400
bottom = 1

totaltime = 300 * 60
dt = 1
ElapsedTime = 0
tolh = 10 ** -6

# # Mesh Genetation
nxc = 400
nxv = nxc + 1
dx = L / nxc
xv = np.zeros(nxc)
for i in list(range(nxc - 1)):
    xv[i + 1] = xv[i] + dx

xc = np.zeros(nxc)
for i in list(range(nxc - 1)):
    xc[i] = 0.5 * (xv[i + 1] + xv[i])

# # Initial Conditions
Vn1 = np.zeros(nxc)
Vn = np.zeros(nxc)

Hn1 = np.zeros(nxc)
Hn = np.zeros(nxc)

Q = np.zeros(totaltime)


def a_trap(b, ss, h):
    area = h * (b + h * ss)
    return area


def p_trap(b, ss, h):
    perimeter = b + 2 * h * sqrt(1 + ss ** 2)
    return perimeter


def t_trap(b, ss, h):
    freesurf = b + 2 * h * ss
    return freesurf


for k in list(range(1, 4)):
    j = 0
    while ElapsedTime < totaltime:
        ElapsedTime = ElapsedTime + dt
        Hn1[0] = HnL * dt
        Vn1[0] = (1 / n) * (((a_trap(bottom, sideslope, Hn1[0])) /
                             p_trap(bottom, sideslope, Hn[0])) ** (2 / 3)) * sqrt(S0)
        if ElapsedTime > Tl:
            HnL = 0
        for i in list(range(1, nxc - 1)):
            d_str = 0.5 * (a_trap(bottom, sideslope, Hn[i + 1]) / p_trap(bottom, sideslope, Hn[i + 1]) +
                           a_trap(bottom, sideslope, Hn[i - 1]) / p_trap(bottom, sideslope, Hn[i - 1]))
            v_str = 0.5 * (Vn[i + 1] + Vn[i - 1])
            Hn1[i] = 0.5 * (Hn[i + 1] + Hn[i - 1]) - 0.5 * (dt / dx) * d_str * (Vn[i + 1] - Vn[i - 1]) \
                - 0.5 * (dt / dx) * v_str * \
                (Hn[i + 1] - Hn[i - 1]) + (HnL * dt)
            if Hn1[i] < tolh:
                Hn1[i] = 0
                Vn1[i] = 0
            else:
                Sf = ((n ** 2) * Vn[i] * abs(Vn[i])) / (
                    (a_trap(bottom, sideslope, Hn1[i]) / p_trap(bottom, sideslope, Hn1[i])) ** (2 / 3))
                Vn1[i] = 0.5 * (Vn[i + 1] + Vn[i - 1]) - 0.5 * (dt / dx) * 9.81 * (Hn[i + 1] - Hn[i - 1]) - \
                    0.5 * (dt / dx) * v_str * \
                    (Vn[i + 1] - Vn[i - 1]) + 9.81 * dt * (S0 - Sf)

        Hn1[nxc - 1] = Hn1[nxc - 2]
        Vn1[nxc - 1] = Vn1[nxc - 2]

        Q[j] = Hn1[nxc - 1] * Vn1[nxc - 1]
        j += 1

        Hn = Hn1
        Vn = Vn1

        if ElapsedTime < totaltime:
            print('Elapsed Time: ', ElapsedTime,
                  ', courant: ', max(Vn1) * dt / dx)
            if ElapsedTime + dt > totaltime:
                dt = totaltime - ElapsedTime

time1 = np.zeros(totaltime)
i = 1
# dt1 = Tl / totaltime
for i in list(range(totaltime)):
    time1[i] = time1[i - 1] + 1

plt.plot(time1, Q)
plt.show()

t1 = time.time()

print('time required: ', (t1 - t0) / 60)
# write the export excel
result = xlwt.Workbook('Uni_project')
# write the export sheet
ws = result.add_sheet('Sheet1')

header = ['Q1', 'time']
# write the header of export excel
for i, header in enumerate(header):
    ws.write(0, i, header)

for i, Q in enumerate(Q):
    ws.write(i + 1, 0, Q)
for i, time1 in enumerate(time1):
    ws.write(i + 1, 1, time1)

result.save('result1.xls')
