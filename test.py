import numpy as np
# L = 2000
# nxc = 400
# nxv = nxc + 1
# dx = L / nxc
# xv = np.zeros(nxc)
# for i in list(range(nxc - 1)):
#     xv[i + 1] = xv[i] + dx

# print(list(range(5)))
# print('\n',range(5))

n=10
# for i in range(n - 1, 0, -1):
#     print(i)

h_n1 = np.reshape([10 for i in np.zeros(n * n)], [n, n])
z = np.ones(n)*20
z[2]=21
print(z[2:])
print(h_n1[[3][:]])
for j in range(1, n):
    for i in range(1, n):
        h_n1[i][j] = z[i]
