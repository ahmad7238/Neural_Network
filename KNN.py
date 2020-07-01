import xlrd
import numpy as np
import xlwt

data = xlrd.open_workbook('Rain&Flowrate.xlsx')
sheet = data.sheet_by_index(0)
x_title = []
y_title = []
for i in range(sheet.ncols):
    if i % 2 == 0:
        x_title.append(sheet.cell(0, i).value)
    else:
        y_title.append(sheet.cell(0, i).value)


data = {}
for i in range(sheet.ncols):
    if i % 2 == 0:
        data['x'] = np.zeros(sheet.nrows-1)
    else:
        data['y'] = np.zeros(sheet.nrows-1)

print('\n\ncount of all data =', len(data['x']))
obs_count = int(input('\nenter the count of observation data: '))

if obs_count >= len(data['x']):
    obs_count = int(
        input('count is extra, enter the count of observation data again = '))


expect_count = int(input('\nenter the count of expected data: '))

if expect_count < 1 or expect_count > len(data['x'])-obs_count:
    expect_count = int(
        input('count is extra, enter the count of expected data again = '))


x_obs = np.zeros(obs_count)
y_obs = np.zeros(obs_count)
x_expected = np.zeros(expect_count)
y_expected = np.zeros(expect_count)

for i in range(sheet.nrows-(expect_count+1)):
    x_obs[i] = sheet.cell(i+1, 0).value
    y_obs[i] = sheet.cell(i+1, 1).value


for i in range(len(x_expected)):
    x_expected[i] = sheet.cell(i+1+obs_count, 0).value
    y_expected[i] = sheet.cell(i+1+obs_count, 1).value

x_obs_reshape = x_obs.reshape(obs_count, 1)
y_obs_reshape = y_obs.reshape(obs_count, 1)

weight_array_test = [1 for i in range(len(x_obs))]


dist = np.zeros(len(x_obs)*expect_count).reshape(expect_count, len(x_obs))

for i in range(expect_count):
    for j in range(len(x_obs)):
        dist[i][j] = abs(weight_array_test[j]*(x_expected[i]-x_obs[j]))

dist_sort = np.sort(dist)


'''k real'''
k_ask = int(input('\nenter k = '))
k = [k_ask for i in range(len(y_obs))]


correspond_yobs_sort = np.zeros(
    len(x_obs)*expect_count).reshape(expect_count, len(x_obs))
correspond_yobs_index_sort = [np.lexsort(
    (y_obs, dist[i])) for i in range(expect_count)]

for i in range(len(correspond_yobs_index_sort)):
    h = 0
    for j in correspond_yobs_index_sort[i]:
        correspond_yobs_sort[i][h] = y_obs[j]
        h += 1


kernel_up = np.zeros(expect_count*k_ask).reshape(expect_count, k_ask)
kernel_down = np.zeros(expect_count)
kernel = np.zeros(expect_count*k_ask).reshape(expect_count, k_ask)
kernel_in_y = np.zeros(expect_count*k_ask).reshape(expect_count, k_ask)
y_eval = np.zeros(expect_count)

for i in range(expect_count):
    for j in range(k_ask):
        kernel_up[i][j] = 1/dist_sort[i][j]
    kernel_down[i] = np.sum(kernel_up[i])
    for h in range(k_ask):
        kernel[i][h] = kernel_up[i][h]/kernel_down[i]
        kernel_in_y[i][h] = kernel[i][h]*correspond_yobs_sort[i][h]
    y_eval[i] = np.sum(kernel_in_y[i])

RMSE = np.zeros(len(y_obs))
NSE = np.zeros(len(y_obs))


RMSE = [np.sqrt(np.mean((y_expected[i]-y_eval[i]))**2)
        for i in range(expect_count)]

NSE = [1-np.sum((y_eval[i]-y_expected[i])**2) /
       np.sum((y_expected[i]-np.mean(y_expected))**2) for i in range(expect_count)]


result = xlwt.Workbook('KNN')
ws = result.add_sheet('Sheet1')

header = ['Y-Expected', 'Y-Eval', 'RMSE', 'NSE']

ws.write(0, 0, 'k = '+str(k_ask))

for i, header in enumerate(header):
    ws.write(1, i, header)
for i, yexp in enumerate(y_expected):
    ws.write(i+2, 0, yexp)
for i, yeval in enumerate(y_eval):
    ws.write(i+2, 1, yeval)
for i, rmse in enumerate(RMSE):
    ws.write(i+2, 2, rmse)
for i, nse in enumerate(NSE):
    ws.write(i+2, 3, nse)

result.save('KNN_result.xls')
