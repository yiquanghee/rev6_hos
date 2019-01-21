from __future__ import print_function
import colorsys
import datetime
# from diffusion_CN import *
#
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import time
from numpy import float, array, fliplr, arange

import config
from generate_data_rev2 import input_matrices, generate_flow_data, \
    generate_storage_parameters
from solver_1d import *
from tvd import *

from sympy.solvers import solve
from sympy import Symbol

dfltsubdir = config.dfltsubdir
sub_dir_name = ""
# sub_dir_name = raw_input("Enter a sub-directory name"
# " if not " + dfltsubdir + ":")
if sub_dir_name is "":
    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, dfltsubdir)
else:
    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, sub_dir_name)
# print path
##############################################################################
# Reading input file
#
##############################################################################

# input file name
# dflt_inoutnm = raw_input("Use default input file name(y/n)? ")
global input_name
dflt_inoutnm = 'y'
if dflt_inoutnm == 'y':
    for dir_entry in os.listdir(path):
        dir_entry_path = os.path.join(path, dir_entry)
        if dir_entry == 'input.txt':
            input_name = dir_entry_path
else:
    dir_entry = eval(raw_input("Input File Name:'name.txt' ==> "))
    input_name = os.path.join(path, dir_entry)


# --check if a string is a float


def isfloat(astr):
    try:
        float(astr)
    except ValueError:
        return False
    return True


# read model parameters
def read_parameters(filename):
    infile1 = open(filename, 'r')
    parameters_list = {}
    for aline in infile1:
        parameter = []
        no_of_strings = 1  # Initialize the no. of strings for parameter name
        split_words = aline.split()
        no_of_words = len(split_words)
        for iw in range(no_of_words):
            if split_words[iw] == "#":
                break
            else:
                if split_words[iw].isdigit():
                    parameter.append(int(split_words[iw]))
                elif isfloat(split_words[iw]):
                    parameter.append(float(split_words[iw]))
                elif split_words[iw].islower():
                    parameter.append(str(split_words[iw]))
                elif split_words[iw].isupper():
                    no_of_strings += 1
                else:
                    parameter.append(split_words[iw])
        if no_of_strings >= 2:
            substance = ''
            for iw in range(no_of_strings - 1):
                if iw != (no_of_strings - 2):
                    substance += split_words[iw] + '_'
                else:
                    substance += split_words[iw]
        else:
            substance = split_words[0]
        parameters_list[substance] = parameter
    infile1.close()
    return parameters_list


parameters = read_parameters(input_name)
# print parameters
# --Defining parameters
type_flow = parameters['TYPE_OF_FLOW'][0]
# 1 = uni-directional flow, 2 = bi-directional flow -------
unit_dist = parameters['UNIT_OF_DISTANCE'][0]  # 1 = meter, 2 = feet
unit_conc = parameters['UNIT_OF_CONCENTRATION'][0]  # 1 = mg/L, 2 = ug/L
dispcoefopt = parameters['DISP_COEFF_OPTION'][0]  # dispcoefopt =model
# dispersion coefficient option, 1 = no change to model disp coeff, 2 = model
# disp coeff reduced for upstream weighting scheme
modelinputdataopt = parameters['MODEL_INPUT_DATA_OPTION'][0]  # 1 = read data
# from input.txt, 2 = generate data in GenerateData.py
modelopt = parameters['MODEL_OPTION'][0]  # 0 = ADE, 1 = transient storage (
# TS) model, 2 = scaling dispersion (SD) model
nuscheme = parameters['NUMERICAL_SCHEME'][0]
# 1 = weighted finite difference, 2 = Crank-Nicolson, 3 = TVD

# --Read experimental data option
expdataopt = parameters['EXPERIMENTAL_DATA_OPT']
# First value: 0 = no experimental data, 1 = experimental data available
# Second text: experimental data file name
# Third value = number of lines to skip
# Fourth value = number of data sets
# The other values = number of data points for each data set

# extract analytical solution name
analsolname = expdataopt[1].split(".")
analsolfilename = analsolname[0].upper()

dt = parameters['TRANSPORT_TIMESTEP'][0]  # unit: second
fdt = parameters['FLOW_TIMESTEP'][0]  # unit: second
tstart = parameters['TIME_START'][0]  # simulation start time - unit: second
# read stream length
strmlength = parameters['STREAM_LENGTH'][0]  # the length of the study segment
nrch = parameters['TOTAL_REACHES'][0]
if expdataopt[0] == 0:
    lstrchno = parameters['LAST_RCH_NO_FOR_SITES'][0]
else:
    lstrchno = parameters['LAST_RCH_NO_FOR_SITES']
    # last reach number for sites
nrchtssolns = parameters['RCH_NO_FOR_TS_SOLNS'][0]
# reach number for time series solutions

# --Storage Parameters
sareasite = parameters['STORAGE_ZONE_AREA']  # storage zone areas for sites
sratesite = parameters['STORAGE_RATE']  # storage rate (beta) for sites
sdecay = parameters['STORAGE_DECAY_COEFF'][0]  # storage decay coefficient
tmin = parameters['TMIN'][0]  # min. net residence time (hour)
r = parameters['R'][0]  # area/ storage area ratio

# dcoef = parameters['dispersion_coefficient'][0]
dcoef = parameters['DISPERSION_COEFFICIENT']
# initial storage concentration
icon = parameters['INITIAL_CONCENTRATION'][0]  # unit: mg/L or ug/L
iscon = parameters['INITIAL_STORAGE_CONC'][0]  # unit: mg/L or ug/L
# read multiple inflow conc boundary conditions
inflowcon = parameters['INFLOW_CONC'][0]  # unit: mg/L or ug/L
outletcon = parameters['OUTLET_CONC'][0]  # unit: mg/L or ug/L
injpar = parameters['INJECTION_PARAMETERS']  # mass and x-sec area
# 1st value: mass in gram and m2/ft2
# 2nd value:

totalfstep = parameters['TOTAL_FLOW_TIMESTEP'][0]  # total flow step
# revise the variable "rvsfstep" by removing '[0]'
# if there are more than one flow reversal
rvsfdata = parameters['REVERSE_FLOW_DATA']
# reverse flow data ------------------------------------------------------------
# 1st value = no. of flow dir. changes
# 2nd values = reverse flow timesteps depending on the 1st value
# 3rd value = flow

omega = parameters['OMEGA'][0]
alpha = parameters['ALPHA'][0]
nusbc = parameters['NUMBER_OF_UPSTREAM_BC'][0]
ndsbc = parameters['NUMBER_OF_DWNSTREAM_BC'][0]
solopt = parameters['ARRAY_SOLVER_OPTION'][0]  # array solver option
# 1 = linalg,
# 2 = scipy linalg spsolve, 3 = Complete LU Factorization Preconditioner,
# 4 = Incomplete LU Factorization

# --for printing
step_print = parameters['SPATIAL_SOL_TIMESTEP']
nopt = parameters['PRINT_TIME_POINTS'][0]
xdsinput = parameters['PRINT_REACHES']

# read x & y limits for graphing
xgraphlimit = parameters['XDOMAIN']
xgraphstep = parameters['XSTEP'][0]
ygraphlimit = parameters['YRANGE']

# -- define print times for spatial solns


# -- define print reaches for temporal solns
xds = []
if xdsinput[0] == 0:  # all reaches will be printed (option 0)
    for i in range(nrch):
        xds.append(i + 1)
else:  # reaches entered will be printed (option 1)
    xdslen = len(xdsinput) - 1
    for i in range(xdslen):
        xds.append(xdsinput[i + 1])
# print xds

# --define dispersion coefficient
sdcoef = zeros(nrch, float)  # specified dispersion coefficient
for i in range(nrch):
    ndispcoef = len(dcoef)
    if ndispcoef == 1:
        sdcoef[i] = dcoef[0]
    else:
        for j in range(ndispcoef):
            if i <= lstrchno[j]:
                sdcoef[i] = dcoef[j]
                break
            else:
                continue

# print sdcoef
# time.sleep(10)

# --define how to get input data
if modelinputdataopt == 1:
    delx = parameters['DELTA_X']  # the first value : 0 = constant distance
    # 1 = variable distance, the others: distance values
    ia = parameters['IA']
    iac_list = parameters['IAC']
    ja = parameters['JA']
    idspflginput = parameters['DISPERSION_FLAG']

    # -- define dispersion flags
    idspflg = []
    if idspflginput[0] == 0:  # all flags will be 1 (option 0)
        for i in range(nrch):
            idspflg.append(1)
    else:  # flags will be entered (option 1)
        for i in range(nrch):
            idspflg.append(idspflginput[i + 1])
            # print idspflg

else:
    delx, ia, iac_list, ja, idspflg = input_matrices(nrch, strmlength,
                                                     expdataopt)

# --pass storage parameters from input to GenerateData_rev1.py or rev2
starea, srate = generate_storage_parameters(
    modelopt, totalfstep, nrch, sareasite, sratesite, lstrchno)

# --calculate total simulation time (sec)
totaltime = totalfstep * fdt

# --calculate T mininum in second for SD model
tmin_sec = tmin * 3600

# --define delta x
del_x = zeros(nrch, float)
if modelinputdataopt == 1:
    # when "input.txt" is used
    for irch in range(nrch):
        if delx[0] == 0:
            del_x[irch] = delx[1]
        else:
            del_x[irch] = delx[irch + 1]
    # print del_x[0]
else:
    # when generate_data_revX.py" is used
    for irch in range(nrch):
        # convert list to array
        del_x[irch] = delx[irch]

# --calculate initial conc based on injection mass
# del_x should be revised if nusbc is larger than 1
# nusbc = no. of upstream boundary conditions
if injpar[0] != 0.0:
    iconinflowim = zeros(nusbc)
    for i in range(nusbc):
        if unit_dist == 1:  # m3 converted to liter (1m3 = 1000L)
            if unit_conc == 1:  # unit: mg/L (1g = 1000mg)
                iconinflowim[i] = injpar[i] / (injpar[nusbc] * del_x[0])
            else:  # unit: ug/L (1g = 1.0E6)
                iconinflowim[i] = 1.0E3 * injpar[i] / (injpar[nusbc]
                                                       * del_x[0])
        else:  # ft3 converted to liter
            if unit_conc == 1:
                iconinflowim[i] = 1.0E3 * injpar[i] / (28.3168 * injpar[nusbc]
                                                       * del_x[0])
            else:
                iconinflowim[i] = 1.0E6 * injpar[i] / (28.3168 * injpar[nusbc]
                                                       * del_x[0])
        # print(iconinflowim[i])

# --define ia, iac, and ja matrices
iac = array(iac_list)
jap = array(ja) - 1  # ja matrix in Python
iap = array(ia) - 1  # ia matrix in Python


# --read 2D array


def read_data(nrow, ncol, infile2):
    lines = infile2.readlines()
    data = zeros((nrow, ncol), float64)
    for ir in range(nrow):
        read_words = lines[ir].split()
        # convert strings to floats
        read_words_flt = map(float, read_words)
        for jr in range(ncol):
            data[ir, jr] = read_words_flt[jr]
    return data


def importdata():
    # input file name
    global input_name0, input_name1, input_name2, input_name3, input_name4
    dflt_inout_name = raw_input("Use default data file names(y/n)? ")
    if dflt_inout_name == 'y':
        for dir_entry_name in os.listdir(path):
            dir_entry_path_impt = os.path.join(path, dir_entry_name)
            if dir_entry_name == 'XArea.txt':
                input_name0 = dir_entry_path_impt
            elif dir_entry_name == 'Flow.txt':
                input_name1 = dir_entry_path_impt
            elif dir_entry_name == 'Inflow.txt':
                input_name2 = dir_entry_path_impt
            elif dir_entry_name == 'InflowConc.txt':
                input_name3 = dir_entry_path_impt
            elif dir_entry_name == 'OutletConc.txt':
                input_name4 = dir_entry_path_impt
            else:
                continue
    else:
        dir_entry0 = eval(raw_input("Input File Name for XS Areas:'name.txt' \
        ==> "))
        input_name0 = os.path.join(path, dir_entry0)
        dir_entry1 = eval(raw_input("Input File Name for Flows:'name.txt' \
        ==> "))
        input_name1 = os.path.join(path, dir_entry1)
        dir_entry2 = eval(raw_input("Input File Name for Inflows: \
        'name.txt' ==> "))
        input_name2 = os.path.join(path, dir_entry2)
        dir_entry3 = eval(raw_input("Input File Name for Inflow \
        Concentrations:'name.txt' ==> "))
        input_name3 = os.path.join(path, dir_entry3)
# The following code is revised for calculating concs for bi-directional flow.
        dir_entry4 = eval(raw_input("Input File Name for Outlet \
        Concentrations:'name.txt' ==> "))
        input_name4 = os.path.join(path, dir_entry4)

    # --read cross-sectional areas
    infile3 = open(input_name0, 'r')
    area_impt = read_data(totalfstep, nrch, infile3)
    infile3.close()

    # --read flow data
    infile3 = open(input_name1, 'r')
    flow_impt = read_data(totalfstep, nrch, infile3)
    infile3.close()
    # --read inflow data
    infile3 = open(input_name2, 'r')
    inflow_impt = read_data(totalfstep, nusbc, infile3)
    infile3.close()
    # --read inflow conc data
    infile3 = open(input_name3, 'r')
    inflowconc_impt = read_data(totalfstep, nusbc, infile3)
    infile3.close()

    return area_impt, flow_impt, inflow_impt, inflowconc_impt


# --open output file
output_entry = 'Results_rev6.txt'
output_name = os.path.join(path, output_entry)
outfile = open(output_name, 'w')

# --open error file
errorfile_entry = 'Errors.txt'
errorfile_name = os.path.join(path, errorfile_entry)
errorfile = open(errorfile_name, 'w')

# --open temporary output file
tempfile_entry = 'TemporaryOutput.txt'
tempfile_name = os.path.join(path, tempfile_entry)
tempfile = open(tempfile_name, 'w')

# --read data from input.txt or generate them using generate_data_revx.py
# read data from input.txt
if modelinputdataopt == 1:
    area, flow, inflow, inflowconc = importdata()
# generate data using generate_data_revx.py
else:
    # if model input data option is not 1,
    # revise "generate_flow_model_data" function -------
    # revision on Jan. 2019 ----------------------------------------------------
    # arguments "rvsfdata" (reverse flow data) added
    # --------------------------------------------------------------------------

    if injpar[0] == 0.0:
        area, flow, inflow, inflowconc = generate_flow_data(
            totalfstep, nrch, nusbc, inflowcon, lstrchno, delx[0], rvsfdata)
    else:
        # define imported initial concentrations
        area, flow, inflow, inflowconc = generate_flow_data(
            totalfstep, nrch, nusbc, iconinflowim, icon, delx[0], rvsfdata)

# set global constants
global cold, scold, told

# determine time level counter and total calculation time
k = 0  # time level counter
totalstep = totalfstep * int(fdt / dt)  # total calculation time
# initialize previouse transport timestep
told = 0.0

# --set downstream boundary conditions
dsbound = zeros(ndsbc)
for i in range(ndsbc):
    dsbound[i] = 0.0  # downstream boundary condition - zero gradient

# --initialize parameters
farea = zeros(nrch, float)  # area at each flow timestep
fflow = zeros(nrch, float)  # flow at each flow timestep
vel = zeros(nrch, float)  # velocity at each flow timestep
gamma = zeros(nrch, float)  # Courant number at each flow timestep
finflow = zeros(nusbc, float)  # inflow at each flow timestep
finflowconc = zeros(nusbc, float)  # inflow concs at each flow timestep

# storage parameters
fsarea = zeros(nrch, float)  # storage area at each flow timestep
fsrate = zeros(nrch, float)  # storage rate (beta) at each flow timestep

# initialize new calculation time and concentration
tc = zeros(totalstep)  # calculation time
cnewcalc = zeros((totalstep, nrch), float)
scnewcalc = zeros((totalstep, nrch), float)
em = zeros(nrch, float)  # model dispersion
print('\n')
print('Courant numbers:\n')

# initialize the number of error messages
errorcount = 0

# initialize the upstream boundary condtion for 'n' timestep,
# inflowconcn, outletconcn ------
finflowconcn = zeros(nusbc, float)

for iusbc in range(nusbc):
    finflowconcn[iusbc] = inflowconc[0, iusbc]

# initialize concs
cnew = zeros(nrch, float)

# for bi-directional flow
if type_flow == 2:
    ci_bidir = zeros(nrch, float)
    # initial condition for bi-directional flow -------
    fflow_rvs = zeros(nrch, float)
    # reversed flow at each flow timestep --------
    farea_rvs = zeros(nrch, float)
    # reversed area at each flow timestep --------

scnew = zeros(nrch, float)

# initialize concs for TVD scheme
cadv = zeros(nrch, float)
cothers = zeros(nrch, float)

# flow timestep starts
for fstep in range(totalfstep):
    for i in range(nusbc):
        if fstep >= (rvsfdata[1] - 1):
            finflow[i] = abs(inflow[fstep, i])  # inflow at each flow timestep
            # print(finflow[i])
        else:
            finflow[i] = inflow[fstep, i]
        finflowconc[i] = inflowconc[fstep, i]  # inflow conc.

    # assign flow data at each flow timestep
    for i in range(nrch):
        farea[i] = area[fstep, i]
        # make flow positive after flow reversal -----------
        # revise if there are more than one flow reversal
        # The 2nd value, rvsfdata[1], is a reverse flow timestep
        # for one flow reversal.
        # However, this value should be updated depending on the order of flow
        # reversal if more than one flow reversal occurs.
        if fstep >= (rvsfdata[1] - 1):
            fflow[i] = abs(flow[fstep, i])
            # print(fflow[i])
        else:
            fflow[i] = flow[fstep, i]
        # storage parameters
        fsarea[i] = starea[fstep, i]
        fsrate[i] = srate[fstep, i]

        if farea[i] != 0.0:
            vel[i] = fflow[i] / farea[i]
        else:
            vel[i] = 0.0

        # calculate Courant number (gamma <1)
        gamma[i] = vel[i] * dt / del_x[i]

        # calculate model dispersion coefficient
        adispersion = Dispersion(path, alpha, nrch, dispcoefopt, sdcoef[i],
                                 i, vel[i], del_x[i])
        em[i], errormessage = adispersion.modeldisp()

        # print error messages
        if errormessage != 'No error!':
            errorfile.write(errormessage)
            errorcount += 1
    if farea[i] != 0.0 and fflow[i] != 0.0:
        print(gamma)
    else:
        print('Warning: The flow and cross-sectional area are zeros for flow '
              'timestep %d.' % fstep)
    # revise below if there are more than one flow reversal
    if (type_flow == 2) and (fstep >= (rvsfdata[1] - 1)):
        fflow_rvs = fliplr([fflow])[0]  # flip flow data
        farea_rvs = fliplr([farea])[0]  # area at each flow timestep

    # transport timestep starts from here
    for tstep in range(int(fdt / dt)):
        # select a numerical scheme
        if nuscheme == 1:
            # revise if there is more than one flow reversal
            if (type_flow == 2) and (fstep >= (rvsfdata[1] - 1)):
                method = WeightedFiniteDiffRvs(dt, em, alpha, omega, nrch, iac,
                                               iap, jap, del_x, farea_rvs,
                                               fflow_rvs, finflow, finflowconc,
                                               finflowconcn, fsarea, fsrate,
                                               dsbound, solopt, idspflg,
                                               sdecay, tmin_sec, modelopt, r)
            else:
                method = WeightedFiniteDiff(dt, em, alpha, omega, nrch, iac,
                                            iap, jap, del_x, farea, fflow,
                                            finflow, finflowconc, finflowconcn,
                                            fsarea, fsrate, dsbound, solopt,
                                            idspflg, sdecay, tmin_sec,
                                            modelopt, r)
        elif nuscheme == 2:
            method = CrankNicolson(dt, em, alpha, omega, nrch, iac, iap,
                                   jap, del_x, farea, fflow, finflow,
                                   finflowconc, finflowconcn, fsarea, fsrate,
                                   dsbound, solopt, idspflg, sdecay, tmin_sec,
                                   modelopt, r)
        # -- set initial or previous conditions
        if fstep == 0 and tstep == 0:
            if farea[i] != 0.0 and fflow[i] != 0.0:
                iconarray = method.set_initial_condition(icon, iscon)
                if nuscheme == 3:
                    cadv = TVDMain(tstep, nrch, gamma, inflowcon, iconarray)
                    cothers, scnew, t = method.solve()
                    # combine concentration by advection and other components
                    for i in range(nrch):
                        cnew[i] = cadv[i] + cothers[i]
                else:
                    cnew, scnew, t = method.solve()
            else:
                pass
                t = told + dt
        elif fstep < (rvsfdata[1] - 1):
            if farea[i] != 0.0 and fflow[i] != 0.0:
                method.set_previous_condition(cold, scold, told)
                if nuscheme == 3:
                    cadv = TVDMain(tstep, nrch, gamma, inflowcon, cold)
                    cothers, scnew, t = method.solve()
                    # combine concentration by advection and other components
                    for i in range(nrch):
                        cnew[i] = cadv[i] + cothers[i]
                else:
                    cnew, scnew, t = method.solve()
                    # set initial condition for reversal flow -------
                    if (fstep == (rvsfdata[1] - 2)) and (tstep == (fdt / dt - 1)):
                        ci_bidir = fliplr([cnew])[0]  # flip cnew array
                        # tempfile.write("   x       C0\n")
                        # for i in range(nrch):
                # tempfile.write("%7.1f  %12.7f\n" % (del_x[i], ci_bidir[i]))
                        # print(ci_bidir[i])
            else:
                pass
                t = told + dt
        else:
            if type_flow == 2:
                if farea_rvs[i] != 0.0 and fflow_rvs[i] != 0.0:
                    method.set_previous_condition(cold, scold, told)
                    if nuscheme == 3:
                        cadv = TVDMain(tstep, nrch, gamma, inflowcon, cold)
                        cothers, scnew, t = method.solve()
                    # combine concentration by advection and other components
                        for i in range(nrch):
                            cnew[i] = cadv[i] + cothers[i]
                    else:
                        cnew, scnew, t = method.solve()
                else:
                    pass
                    t = told + dt
            else:
                if farea[i] != 0.0 and fflow[i] != 0.0:
                    method.set_previous_condition(cold, scold, told)
                    if nuscheme == 3:
                        cadv = TVDMain(tstep, nrch, gamma, inflowcon, cold)
                        cothers, scnew, t = method.solve()
                    # combine concentration by advection and other components
                        for i in range(nrch):
                            cnew[i] = cadv[i] + cothers[i]
                    else:
                        cnew, scnew, t = method.solve()
                else:
                    pass
                    t = told + dt
        # print cnew
        # print t

        tc[k] = t
        for i in range(nrch):
            cnewcalc[k, i] = cnew[i]
            scnewcalc[k, i] = scnew[i]
            # if i == 32:
            #     tempfile.write("  %5d %12.7f\n" % (k, cnewcalc[k, i]))
        # set previous concs and time
        # set previous concs with flipped conc values
        if (type_flow == 2) and (tc[k] == fdt * (rvsfdata[1] - 1)):
            cold = ci_bidir
        else:
            cold = cnew
        scold = scnew
        told = t

        k += 1

# when there is no error or warning
if errorcount == 0:
    errorfile.write('No errors or warnings found in segment lengths.')

##############################################################################
# Writing model output and creating graphs for model output
#
##############################################################################

# -----------------------------------------------------------------------------
# Printing output at a specified time
# -----------------------------------------------------------------------------
outfile.write('Concentration versus Distance:\n')
outfile.write('\n')
outfile.write('        Dist     Time      Num Conc')
outfile.write('\n')
if unit_dist == 1:
    if unit_conc == 1:
        outfile.write('        (m)     (hr)      (mg/L)')
    else:
        outfile.write('        (m)     (hr)      (ug/L)')
else:
    if unit_conc == 1:
        outfile.write('        (ft)     (hr)      (mg/L)')
    else:
        outfile.write('        (ft)     (hr)      (ug/L)')
outfile.write('\n')

xsum = 0.0
xdsm1 = [xds[i] - 1 for i in range(len(xds))]  # reaches to print results
x = zeros(len(xdsm1))
cnewcalc_space = zeros(len(xdsm1), float)
ci_bidir_space = zeros(len(xdsm1), float)

i = 0
for j in xdsm1:
    if j == 0:
        x[i] = del_x[j] / 2
        xsum += del_x[j]
    else:
        x[i] = xsum + del_x[j] / 2
        xsum += del_x[j]
    # print step_print
    tspec = (tstart + tc[step_print]) / 3600.0
    cnewcalc_space[i] = cnewcalc[step_print, j]
    if type_flow == 2:
        ci_bidir_space[i] = ci_bidir[j]
    outfile.write("  %10.2f  %7.2f  %12.7f  %12.7f" % (x[i], tspec,
                                                       cnewcalc_space[i],
                                                       ci_bidir_space[i]))
    outfile.write('\n')
    i += 1

print("\n The numerical solution is in space: \n", cnewcalc_space)
# -----------------------------------------------------------------------------
# Printing output at a specified distance
# -----------------------------------------------------------------------------
ts_print = zeros(nopt)
stepcoef = len(tc) / nopt - 1
cnewcalc_time = zeros(nopt, float)
scnewcalc_time = zeros(nopt, float)
outfile.write('\n')
outfile.write('Concentration Time Series:\n')

for i in range(nrch):
    outfile.write('\n')
    outfile.write('   Reach %2d\n' % (i + 1))
    outfile.write('        Time      Num Conc')
    outfile.write('\n')
    if unit_conc == 1:
        outfile.write('        (hr)      (mg/L)')
    else:
        outfile.write('        (hr)      (ug/L)')
    outfile.write('\n')
    # a reach no "i" for forward flow is equivalent to
    # "lstrchno - i" for reverse flow.
    irvs = int(lstrchno[0]) - (i + 1)
    for j in range(nopt):
        ts_print[j] = (tstart + tc[stepcoef * j + j]) / 3600.0
        if ts_print[j] >= (fdt * (rvsfdata[1] - 1) / 3600):
            # for reverse flow
            cnewcalc_time[j] = cnewcalc[stepcoef * j + j, irvs]
            scnewcalc_time[j] = scnewcalc[stepcoef * j + j, irvs]
        else:
            # for forward flow
            cnewcalc_time[j] = cnewcalc[stepcoef * j + j, i]
            scnewcalc_time[j] = scnewcalc[stepcoef * j + j, i]
        outfile.write("  %10.3f  %12.7f   %12.7f" % (
            ts_print[j], cnewcalc_time[j], scnewcalc_time[j]))
        outfile.write('\n')
    # if i == 16:
    # if i == 2:
    if i == nrchtssolns - 1:
        plt.plot(ts_print, cnewcalc_time, label=i + 1)
        plt.legend(loc='upper right', ncol=1)
        plt.title('Concentration vs Time')
        plt.xlabel('t (hr)')
        # plt.xlim(0, 1.0)
        # plt.xlim(0, 20)
        plt.xlim(xgraphlimit[0], xgraphlimit[1])
        plt.ylim(ygraphlimit[0], ygraphlimit[1])
        # plt.ylim(0, 0.7)
        if unit_conc == 1:
            plt.ylabel('c (mg/L)')
        else:
            plt.ylabel('c (ug/L)')
            # plt.ylim(0,0.18)
    print("\nReach:", i + 1)
    print("The numerical solution in time is:\n", cnewcalc_time)
pp = PdfPages(os.path.join(path, 'ConcTS.pdf'))
pp.savefig()
pp.close()
plt.close()


# -----------------------------------------------------------------------------
# Creating a conc. time series graph for the Monocacy River, Case 5
# at four sites
# -----------------------------------------------------------------------------


def get_color(color, cvalue):
    for hue in range(color):
        hue = 1. * hue / color
        col = [int(xi) for xi in colorsys.hsv_to_rgb(hue, 1.0, cvalue)]
        yield "#{0:02x}{1:02x}{2:02x}".format(*col)


if expdataopt[0] == 1:
    # --Read experimental data
    #  expdata_dir_entry = eval(raw_input("Input File Name for Experimental \
    # Data:'name.txt' ==> "))
    # expdata_dir_entry = 'Experimental_Data.txt'
    expdata_dir_entry = expdataopt[1]
    expdata_input_name = os.path.join(path, expdata_dir_entry)
    infile = open(expdata_input_name, 'r')
    # lines = infile.readlines()

    no_of_sites = expdataopt[3]

    color1 = get_color(no_of_sites, 230)
    color2 = get_color(no_of_sites, 100)

    # Assign time and conc. data to each site.
    # lstrchno_m1 = array(lstrchno) - 1
    # OTIS Application 4
    # lstrchno_m1 = 456  # reach number = 457 since python starts from zero
    # not one

    for i in range(no_of_sites):
        isite = 0
        time_exp = []
        conc_exp = []
        sitename = expdataopt[i + no_of_sites + 4].upper()
        # skip header rows in an experimental data file
        if i == 0:
            for _ in xrange(expdataopt[2]):
                next(infile)
        for line in infile:
            words = line.split()
            time_exp.append(words[0])
            conc_exp.append(words[1])
            isite += 1
            if isite == expdataopt[i + 4]:
                break

        # define no. of experimental data/analytical solutions
        nexpdata = expdataopt[4]
        # print (nexpdata)

        # define zero arrays for time and concentration data
        time_exp_ary = zeros(nexpdata)
        conc_exp_ary = zeros(nexpdata)

        # convert lists of analytical solution/experimental data to arrays
        for i in range(nexpdata):
            time_exp_ary[i] = time_exp[i]
            conc_exp_ary[i] = conc_exp[i]

        acolor = next(color1)

        # plt.plot(time_exp_ary, conc_exp_ary, color=acolor, marker='o',
        #          linestyle='None', label='experimental/analytical - %s'
        #                                  % sitename)
        plt.scatter(time_exp_ary, conc_exp_ary, color=acolor, marker='o',
                    s=4, label='experimental/analytical-%s' % sitename)
        # s is marker size

        bcolor = next(color2)

        # The following code is to calculate an equivalent reach number
        # for reverse flow

        # calculate a distance value for time series solutions for forward flow
        disfwd = int(delx[1]) * (nrchtssolns - 1) + int(delx[1]) / 2

        # calculate a distance value for time series solutions for reverse flow
        disrvs = int(delx[1]) * nrch - disfwd

        # use the solvers module in SymPy to determine the equivalent
        # calculate a equivalent reach number for reverse flow
        xrvs = Symbol('xrvs')
        nrchtssolns_rvs = solve((xrvs-1) * int(delx[1])
                                + int(delx[1]) / 2 - disrvs, xrvs)
        nrchtssolns_rvs_int = int(nrchtssolns_rvs[0]) - 1
        # nrchtssolns_rvs_int = 36

        # initialize time and concentration for analytical and numerical solns
        ts_print1 = zeros(nopt)
        cnewcalc_time1 = zeros(nopt, float)

        for j in range(nopt):
            ts_print1[j] = (tstart + tc[stepcoef * j + j]) / 3600.0
            if ts_print1[j] >= (fdt * (rvsfdata[1] - 1) / 3600):
                # for reverse flow
                cnewcalc_time1[j] = cnewcalc[stepcoef * j + j,
                                             nrchtssolns_rvs_int]
            else:
                # for forward flow
                cnewcalc_time1[j] = cnewcalc[stepcoef * j + j, nrchtssolns - 1]
        plt.plot(ts_print1, cnewcalc_time1, color=bcolor, marker='x',
                 label='numerical-%s' % sitename)

        plt.legend(loc='upper right', ncol=1, fontsize='small')
        # print(ts_print, cnewcalc_time1)
        # extract project tile from experimental data info
        expdatatitle = expdataopt[1].split(".")
        projtitle = expdatatitle[0].upper()
        if projtitle == 'OBS457':
            if modelopt == 0:
                plt.title('Conc Time Series for Huey Creek Lithium, '
                          'OTIS Application 4 \n x = 457 meters')

            elif modelopt == 1:
                plt.title('Conc Time Series for Huey Creek Lithium, '
                          'OTIS Application 4\n x = 457 meters\n'
                          'Transient storage model with storage zone x-sect. '
                          'area of %.1f,\n %.1f, %.1f, %.1f SM and exchange '
                          'rate of %2.1E, %2.1E, %2.1E, %2.1E /s '
                          % (sareasite[0], sareasite[1], sareasite[2],
                             sareasite[3], sratesite[0], sratesite[1],
                             sratesite[2], sratesite[3]), fontsize=8)
            else:
                plt.title('Conc Time Series for Huey Creek Lithium, '
                          'OTIS Application 4 \n x = 457 meters\n'
                          'Scaling dispersion model with x-sect. area ratio of'
                          '%2.1E,\n and Tmin of %2.1E hours'
                          % (r, tmin), fontsize=8)

        elif projtitle == analsolfilename:
            plt.title('Conc Time Series for Numerical & Analytical Solutions')
        else:
            if modelopt == 0:
                plt.title('Conc Time Series for the Monocacy River, '
                          'Case-5-Central in Space-disp coeff of '
                          '%d, %d, %d, %d \n and initial inflow conc of '
                          '%.1f ug/L-Advection-dispersion equation'
                          % (dcoef[0], dcoef[1], dcoef[2], dcoef[3],
                             inflowcon[0]), fontsize=8)
            elif modelopt == 1:
                plt.title('Conc Time Series for the Monocacy River, '
                          'Case-5-Central in Space-disp coeff of '
                          '%d, %d, %d, %d \n and initial inflow conc of '
                          '%.1f ug/L-Transient Storage model with storage zone'
                          ' x-sect. area of %.1f,\n %.1f, %.1f, %.1f SF and '
                          'exchange rate of %2.1E, %2.1E, %2.1E, %2.1E /s '
                          % (dcoef[0], dcoef[1], dcoef[2], dcoef[3],
                             inflowcon[0], sareasite[0], sareasite[1],
                             sareasite[2], sareasite[3], sratesite[0],
                             sratesite[1], sratesite[2],
                             sratesite[3]), fontsize=8)
            else:
                plt.title('Conc Time Series for the Monocacy River, '
                          'Case-5-Central in Space-disp coeff of '
                          '%d, %d, %d, %d \n and initial inflow conc of '
                          '%.1f ug/L-Scaling dispersion model with x-sect. '
                          'area ratio of %2.1E,\n and Tmin of %2.1E hours'
                          % (dcoef[0], dcoef[1], dcoef[2], dcoef[3],
                             inflowcon[0], r, tmin), fontsize=8)

        plt.xlabel('t (hr)')
        if unit_conc == 1:
            plt.ylabel('c (mg/L)')
        else:
            plt.ylabel('c (ug/L)')
        if projtitle == 'OBS457':
            plt.xlim(10, 20)
            plt.ylim(0, 3)
        # set x and y axis ranges
        # if analytical solutions are used for experimental data
        elif projtitle == analsolfilename:
            # plt.xlim(0, 20)
            # plt.xlim(0, 1.0)
            # plt.ylim(0, 0.7)
            plt.xlim(xgraphlimit[0], xgraphlimit[1])
            plt.ylim(ygraphlimit[0], ygraphlimit[1])
            plt.xticks(arange(xgraphlimit[0], xgraphlimit[1]+0.1,
                              step=xgraphstep))
        else:
            plt.xlim(0, 70)
            plt.ylim(0, 16)

    # superimpose grid
    plt.grid()

    if projtitle == 'OBS457':
        pp1 = PdfPages(os.path.join(path, 'OTIS_App4_ConcTS.pdf'))
    elif projtitle == analsolfilename:
        pp1 = PdfPages(os.path.join(path, 'Analytical_ConcTS.pdf'))
    else:
        pp1 = PdfPages(os.path.join(path, 'Monocacy_ConcTS.pdf'))

    pp1.savefig()
    pp1.close()
    plt.close()
    infile.close()

# closing files

outfile.close()
errorfile.close()

# -----------------------------------------------------------------------------
# Creating graphs
#
# -----------------------------------------------------------------------------
with PdfPages(os.path.join(path, 'Conc_Distance_Graphs.pdf')) as pdf:
    plt.figure(figsize=(10, 8))
    plt.plot(x, cnewcalc_space, 'r-o')
    plt.title('Concentration vs Distance')
    if unit_dist == 1:
        plt.xlabel('x (m)')
        if unit_conc == 1:
            plt.ylabel('c (mg/L)')
        else:
            plt.ylabel('c (ug/L)')
    else:
        plt.xlabel('x (ft)')
        if unit_conc == 1:
            plt.ylabel('c (mg/L)')
        else:
            plt.ylabel('c (ug/L)')
    pdf.savefig()
    plt.close()

    d = pdf.infodict()
    d['Title'] = 'Concentration Distance Graphs'
    d['Author'] = u'Quanghee Yi\xe4nen'
    d['Subject'] = 'Concentration graphs for SWTSample01 by SWT_python code'
    d['Keywords'] = 'PdfPages conc. graphs keywords author title subject'
    d['CreationDate'] = datetime.datetime(2014, 11, 19)
    d['ModDate'] = datetime.datetime.today()

# raw_input("\nPress return to exit")
