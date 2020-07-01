import xlrd
import numpy as np
from numpy import mean
from scipy.stats import linregress, ttest_ind, ttest_ind_from_stats, ttest_rel
import matplotlib.pyplot as plt

data = xlrd.open_workbook('Rain&Flowrate.xlsx')
sheet = data.sheet_by_index(0)
regression_rain = []
regression_flowrate = []
for i in range(sheet.ncols):
    if i % 2 == 0:
        regression_rain.append(sheet.cell(0, i).value)
    else:
        regression_flowrate.append(sheet.cell(0, i).value)


data = {}
h = 0
k = 0
for i in range(sheet.ncols):
    if i % 2 == 0:
        data['x'+str(k)] = np.zeros(sheet.nrows-1)
        k += 1
    else:
        data['y'+str(h)] = np.zeros(sheet.nrows-1)
        h += 1

h = 0
k = -1
for i in range(sheet.ncols):
    for j in range(sheet.nrows-1):
        if i % 2 == 0:
            data['x'+str(int(h))][j] = sheet.cell(j+1, i).value
        else:
            data['y'+str(int(k+0.5))][j] = sheet.cell(j+1, i).value
    k += 0.5
    h += 0.5

mean = {}
n = {}
h = 0
k = 0
for i in range(sheet.ncols):
    if i % 2 == 0:
        mean['x'+str(h)] = np.mean(data['x'+str(h)])
        n['x'+str(h)] = np.size(data['x'+str(h)])
        h += 1
    else:
        mean['y'+str(k)] = np.mean(data['y'+str(k)])
        n['y'+str(k)] = np.size(data['y'+str(k)])
        k += 1


cov_xy = {}
std_x = {}
std_y = {}
r = {}
b = {}
a = {}
p = {}
limited_regression = {}
xp = {}
for i in range(int(sheet.ncols/2)):
    cov_xy['cov_xy'+str(i)] = np.sum((data['x'+str(i)] -
                                      mean['x'+str(i)])*(data['y'+str(i)]-mean['y'+str(i)]))
    std_x['std_x'+str(i)] = np.sqrt(np.sum((data['x' +
                                                 str(i)]-mean['x'+str(i)])**2))
    std_y['std_y'+str(i)] = np.sqrt(np.sum((data['y' +
                                                 str(i)]-mean['y'+str(i)])**2))
    r['r'+str(i)] = cov_xy['cov_xy'+str(i)] / \
        (std_x['std_x'+str(i)]*std_y['std_y'+str(i)])
    b['b'+str(i)] = cov_xy['cov_xy'+str(i)]/(std_x['std_x'+str(i)]**2)
    a['a'+str(i)] = mean['y'+str(i)]-(b['b'+str(i)]*mean['x'+str(i)])

    print('\nRegression Coeffienct'+str(i), ' =', r['r'+str(i)])
    print('Regression Equation: y'+str(i), ' =',
          a['a'+str(i)], ' +(', b['b'+str(i)], ')x')
    p['p'+str(i)] = np.polyfit(data['x'+str(i)], data['y'+str(i)], 3)
    limited_regression = ttest_ind(data['y'+str(0)], data['y'+str(i)])
    print('p_value'+regression_rain[i], ': ', limited_regression[1])
    if limited_regression[1] < 0.025:
        print('accept the null hypothesis of ' +
              regression_flowrate[0], 'and', regression_flowrate[i])
    else:
        print('reject the null hypothesis of ' +
              regression_flowrate[0], 'and', regression_flowrate[i])
    xp['xp'+str(i)] = np.linspace(min(data['x'+str(i)]),
                                  max(data['x'+str(i)]), 100)
    plt.plot(data['x'+str(i)], data['y'+str(i)], 'o', label='data')
    plt.plot(xp['xp'+str(i)], np.polyval(p['p'+str(i)],
                                         xp['xp'+str(i)]), '--', label='fit')
    plt.title('regression_'+regression_rain[i])
    plt.legend()
    plt.show()
    print('\npress Enter')
    input()
