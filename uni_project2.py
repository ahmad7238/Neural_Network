# # Title: Class Project(Problem1)
#   Assumptions: Prismatic channel
#                fix lateral flow
#                Rough bed (By only adding tolh to initial condition)
#                Trapezoidal cross-section

# settings
from math import sqrt, tan, radians
import matplotlib.pyplot as plt
import numpy as np
import xlwt
import time

t0 = time.time()

S0 = 0.001
sideslope = tan(radians(60))
n_man = 0.02
L = 100
bottom = 0.2

totaltime = 300
dt = 0.02
elapsedtime = 0
tolh = 1e-06
inf_time = 0

## secinline parameters
maxiter = 100
xnew = np.zeros(maxiter)
Iter = np.zeros(maxiter)
err = np.zeros(maxiter)
xvold = 0.03
xold = 0.08
max_err = 0.001
xnew[0] = xold
Iter[0] = 1
err[0] = 0

Q = 0.007
f0 = 0.1 / 3600000
k = 0.0097 / 60
alpha = 0.617
line_space = 1

# # Mesh Generation
nxc = 200
nxv = nxc + 1
dx = L / nxc
xc = np.zeros(nxc)
elec = np.zeros(nxc)
xv = [(i - 1) + dx for i in list(range(nxv))]
elev = [(5 - S0 * i) for i in xv]
for i in list(range(nxc)):
    xc[i] = 0.5 * (xv[i + 1] + xv[i])
    elec[i] = 0.5 * (elev[i + 1] + elev[i])

# # Initial Conditions
vn1 = np.zeros(nxc)
vn = np.zeros(nxc)

hn1 = np.zeros(nxc)
hn = [tolh for i in list(range(nxc))]

h = np.zeros(int(totaltime/dt)+1)
time1 = np.zeros(int(totaltime/dt)+1)


def a_trap(b, ss, h):
    area = h * (b + h * ss)
    return area


def p_trap(b, ss, h):
    perimeter = b + 2 * h * sqrt(1 + ss ** 2)
    return perimeter


def t_trap(b, ss, h):
    freesurf = b + 2 * h * ss
    return freesurf


def k_l(h_c, t, alph, f, w):
    infiltration = (h_c * (t ** alph) + f * t) / w
    return infiltration


def fsec(x, ss, n, b, s, q):
    y = (ss * (x ** 2) + bottom * x) ** (5 / 3) - (q * n / sqrt(s)) * (
            b + 2 * x * sqrt(1 + ss ** 2)) ** (2 / 3)
    return y


def secinline(x_vold, x_old):
    for iter in list(range(1, maxiter)):
        xnew[iter] = x_old - (fsec(xold, sideslope, n_man, bottom, S0, Q) * (x_vold - x_old) / (
                fsec(xvold, sideslope, n_man, bottom, S0, Q) - fsec(xold, sideslope, n_man, bottom, S0, Q)))
        err[iter] = abs((xnew[iter] - x_old) / xnew[iter]) * 100
        Iter[iter] = iter
        if err[iter] < max_err:
            break
        x_vold = x_old
        x_old = xnew[iter]
    y = xnew[iter]
    return y


dt1 = (totaltime/dt) / nxc
for i in list(range(1, int(totaltime/dt)+1)):
    time1[i] = time1[i - 1] + dt

j = 0
while elapsedtime < totaltime:
    elapsedtime = elapsedtime + dt
    hn1[0] = secinline(xvold, xold)
    vn1[0] = Q / a_trap(bottom, sideslope, hn1[0])
    for i in list(range(1, nxc - 1)):
        d_str = 0.5 * (a_trap(bottom, sideslope, hn[i + 1]) / p_trap(bottom, sideslope, hn[i + 1]) +
                       a_trap(bottom, sideslope, hn[i - 1]) / p_trap(bottom, sideslope, hn[i - 1]))
        v_str = 0.5 * (vn[i + 1] + vn[i - 1])
        hn1[i] = 0.5 * (hn[i + 1] + hn[i - 1]) - 0.5 * (dt / dx) * d_str * (vn[i + 1] - vn[i - 1]) \
                 - 0.5 * (dt / dx) * v_str * (hn[i + 1] - hn[i - 1])
        if hn1[i] > 0.009:
            hn1[i] = hn1[i] - k_l(k, inf_time, alpha, f0, line_space)
        Sf = ((n_man ** 2) * vn1[i] * abs(vn1[i])) / (
                (a_trap(bottom, sideslope, hn1[i]) / p_trap(bottom, sideslope, hn1[i])) ** (2 / 3))
        vn1[i] = 0.5 * (vn[i + 1] + vn[i - 1]) - 0.5 * (dt / dx) * 9.81 * (hn[i + 1] - hn[i - 1]) \
                 - 0.5 * (dt / dx) * v_str * (vn[i + 1] - vn[i - 1]) + 9.81 * dt * (S0 - Sf)

    inf_time += dt

    hn1[nxc - 1] = hn1[nxc - 2]
    vn1[nxc - 1] = vn1[nxc - 2]

    hn = hn1
    vn = vn1

    h[j] = hn[nxc - 1]
    j += 1

    if elapsedtime < totaltime:
        print('Elapsed Time: ', elapsedtime, ', courant: ', max(vn1) * dt / dx)
        if elapsedtime + dt > totaltime:
            dt = totaltime - elapsedtime

t1 = time.time()
print('time required: ', (t1 - t0) / 60)

fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(xv, elev, 'g', xc, hn1 + elec, 'b')
plt.title('water level')
plt.xlabel('Length')
plt.ylabel('hn')

plt.subplot(2, 1, 2)
plt.plot(time1, h)
plt.title('advance phase')
plt.xlabel('time')
plt.ylabel('h')

fig.savefig('Uni_project2.png')
plt.show()

# write the export excel
result = xlwt.Workbook('Uni_project')
# write the export sheet
ws = result.add_sheet('Sheet1')

header = ['t', 'h', 'hn']
# write the header of export excel
for i, header in enumerate(header):
    ws.write(0, i, header)

for i, time1 in enumerate(time1):
    ws.write(i + 1, 0, time1)
for i, h in enumerate(h):
    ws.write(i + 1, 1, h)
for i, hn in enumerate(hn):
    ws.write(i + 1, 2, hn)

result.save('Uni_project2.xls')
