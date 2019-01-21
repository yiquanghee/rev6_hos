from numpy import *
import os
import config
import scipy.interpolate


def input_matrices(nrch, strmlength, expdataopt):
    """This function creates delta x, ia, iac, and ja matrices, and
    dispersion flags.
    """

    global nrchsite
    dfltsubdir = config.dfltsubdir
    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, dfltsubdir)

    # generate delta x
    if expdataopt[0] == 0:
        #  synthetic example 2
        delx = []
        strlength = 600.0
        for i in range(nrch):
            delx.append(strlength / nrch)
    else:
        if strmlength != 0.0:
            delx = []
            # create equally-spaced delta x
            for i in range(nrch):
                delx.append(strmlength / nrch)
        else:
            # case for 114 reaches
            if nrch == 114:
                nrchsite = [28, 24, 30, 28]  # no. of reaches for 4 sites
                rchlength = [1200, 1081, 909, 863]  # reach length
                lstrchlength = [384, 900, 900, 801]  # last reach length
            elif nrch == 427:
                nrchsite = [111, 91, 107, 114]  # no. of reaches for 4 sites - 1
                rchlength = [304, 289, 257, 214]  # reach length
                lstrchlength = [300, 280, 250, 210]  # last reach length
            elif nrch == 38:
                nrchsite = [9, 9, 9, 11]  # no. of reaches for 4 sites - 1
                rchlength = [80, 80, 80, 80]  # reach length
                lstrchlength = [80, 80, 80, 80]  # last reach length
            elif nrch == 504:
                nrchsite = [188, 91, 107, 114]
                # (no. of reaches for 4 sites) - 1
                rchlength = [179, 289, 258, 214]  # reach length
                lstrchlength = [170, 280, 250, 210]  # last reach length
            # case for 4384 reaches
            elif nrch == 4384:
                nrchsite = [1126, 977, 1108, 1169]
                rchlength = [30, 27, 25, 21]  # reach length
                lstrchlength = [24, 26, 20, 20]  # last reach length
            # OTIS application 4
            elif nrch == 650:
                nrchsite = [9, 204, 244, 193]
                # no. of reaches (segments in OTIS)
                rchlength = [1, 1, 1, 1]  # reach length
                lstrchlength = [1, 1, 1, 1]  # last reach length
                # The variable lstrchlength is defined
                # even if all reach lengths are 1.
            else:
                print ('Warning: Enter reach information')

            trchsite = 0
            tsubrchsite = []
            for i in range(4):
                trchsite += nrchsite[i] + 1
                tsubrchsite.append(trchsite - 1)
            # print trchsite
            # print tsubrchsite

            delx = []
            for i in range(trchsite):
                if i < tsubrchsite[0]:
                    delx.append(rchlength[0])
                elif i == tsubrchsite[0]:
                    delx.append(lstrchlength[0])
                elif tsubrchsite[0] < i < tsubrchsite[1]:
                    delx.append(rchlength[1])
                elif i == tsubrchsite[1]:
                    delx.append(lstrchlength[1])
                elif tsubrchsite[1] < i < tsubrchsite[2]:
                    delx.append(rchlength[2])
                elif i == tsubrchsite[2]:
                    delx.append(lstrchlength[2])
                elif tsubrchsite[2] < i < tsubrchsite[3]:
                    delx.append(rchlength[3])
                else:
                    delx.append(lstrchlength[3])

    # generate ia matrix
    ia = []
    add_value = 3
    ia.append(1)
    for i in xrange(nrch):
        if i != nrch - 1:
            ia.append(add_value * (i + 1))
        else:
            ia.append(add_value * (i + 1) - 1)
    # print ia

    # generate iac matrix
    iac = []
    for i in range(nrch):
        if i == 0 or i == nrch - 1:
            iac.append(2)
        else:
            iac.append(3)

    # generate ja matrix
    no_of_ja = 3 * (nrch - 2) + 4
    # ja = zeros(no_of_ja, int)
    ja = []
    for i in xrange(nrch):
        if i == 0:
            ja.append(i + 1)
            ja.append(i + 2)
        elif i == nrch - 1:
            ja.append(i + 1)
            ja.append(i)
        else:
            ja.append(i + 1)
            ja.append(i)
            ja.append(i + 2)

    # generate dispersion flag
    dispflg = []
    for i in range(nrch):
        dispflg.append(1)

    # create output file
    output_entry = 'Input_matrices.txt'
    output_name = os.path.join(path, output_entry)
    outfile = open(output_name, 'w')
    # print delta x
    outfile.write("Delta x:\n")
    if expdataopt[0] == 0:
        for i in xrange(nrch):
            outfile.write("%d " % delx[i])
        outfile.write('\n')
    else:
        # revised for analytical solutions (the first if statement below)
        if strmlength != 0.0:
            for i in xrange(nrch):
                outfile.write("%d " % delx[i])
            outfile.write('\n')
        else:
            for i in xrange(trchsite):
                outfile.write("%d " % delx[i])
            outfile.write('\n')

    # print ia matrix
    outfile.write("IA Matrix Elements:\n")
    for i in xrange(nrch + 1):
        outfile.write("%d " % ia[i])
    outfile.write('\n')

    # print iac matrix
    outfile.write("IAC Matrix Elements:\n")
    for i in xrange(nrch):
        outfile.write("%d " % iac[i])
    outfile.write('\n')

    # print ja matrix
    outfile.write("JA Matrix Elements:\n")
    for i in xrange(no_of_ja):
        outfile.write("%d " % ja[i])
    outfile.write('\n')

    # print dispersion flag
    outfile.write("Dispersion Flags:\n")
    for i in xrange(nrch):
        outfile.write("%d " % dispflg[i])

    outfile.close()
    return delx, ia, iac, ja, dispflg


def calcx(nrch, del_x):
    """ calculate x values which are centers of all the cells.
    """
    xc = zeros(nrch, float)
    for k in range(nrch):
        if k == 0:
            xc[k] = del_x / 2.0
        else:
            xc[k] = k * del_x + del_x / 2.0
    return xc


def read_data_otis(nrow, infilefn):
    otis_lines = infilefn.readlines()
    flow_otis = []
    area_otis = []
    count_num = 0
    count_unsteady = 0
    for ir in range(nrow):
        read_words = otis_lines[ir].split()
        if otis_lines[ir].startswith("#"):
            continue
        elif count_num == 0:
            qstep = float(read_words[0])
            count_num += 1
        elif count_num == 1:
            nflow = int(read_words[0])
            flowloc = zeros(nflow, float)
            count_flowloc = 0
            count_num += 1
        elif 1 < count_num <= 1 + nflow:
            flowloc[count_flowloc] = read_words[0]
            count_flowloc += 1
            count_num += 1
        else:
            if count_unsteady == 0:
                count_unsteady += 1
                continue
            elif count_unsteady == 1:
                for i in range(nflow):
                    flow_otis.append(float(read_words[i]))
                count_unsteady += 1
            elif count_unsteady == 2:
                for i in range(nflow):
                    area_otis.append(float(read_words[i]))
                count_unsteady += 1
            else:
                count_unsteady = 0
    ntimes = len(flow_otis) / nflow  # number of flow timesteps
    # convert flows and areas into arrays
    flow_otis_array = zeros((ntimes, nflow), float64)
    area_otis_array = zeros((ntimes, nflow), float64)
    for it in range(ntimes):
        for ix in range(nflow):
            flow_otis_array[it, ix] = flow_otis[it * nflow + ix]
            area_otis_array[it, ix] = area_otis[it * nflow + ix]

    return qstep, ntimes, nflow, flowloc, flow_otis_array, area_otis_array


def generate_flow_model_data(totalfstep, nrch, nusbc, iconinflow, lstrchno,
                             delx):
    """ This function creates cross-sectional area, flow, inflow, and inflow
    concentration data.
    """
    # set directory for output files
    dfltsubdir = config.dfltsubdir
    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, dfltsubdir)

    # create output file
    output_entry1 = 'XArea_4review.txt'
    output_name1 = os.path.join(path, output_entry1)
    outfile1 = open(output_name1, 'w')
    output_entry2 = 'Flow_4review.txt'
    output_name2 = os.path.join(path, output_entry2)
    outfile2 = open(output_name2, 'w')
    output_entry3 = 'Inflow_4review.txt'
    output_name3 = os.path.join(path, output_entry3)
    outfile3 = open(output_name3, 'w')
    output_entry4 = 'InflowConc_4review.txt'
    output_name4 = os.path.join(path, output_entry4)
    outfile4 = open(output_name4, 'w')

    # set the last reach number for each site
    # case for 114 reaches
    if nrch == 114:
        lstrchno = [29, 54, 85, 114]  # last reach number
    elif nrch == 427:
        lstrchno = [112, 204, 312, 427]  # last reach number
    # case for 4384 reaches
    elif nrch == 4384:
        lstrchno = [1127, 2105, 3215, 4384]  # last reach number
    else:
        if lstrchno is None:
            print ('Warning: Enter last reach numbers for each site')

    # generate cross-sectional area data
    # xareavalue = 289.6
    # Monocacy River Case 5
    # flow_data_title = 'Monocacy River Case 5'
    flow_data_title = 'OTIS Application 4'
    if flow_data_title == 'Monocacy River Case 5':
        xareasite = [253.0, 286.4, 306.8, 395.9]
        xarea = zeros((totalfstep, nrch), float64)
        for ifstep in range(totalfstep):
            for i in range(nrch):
                if i < lstrchno[0]:
                    xarea[ifstep, i] = xareasite[0]
                elif lstrchno[0] <= i < lstrchno[1]:
                    xarea[ifstep, i] = xareasite[1]
                elif lstrchno[1] <= i < lstrchno[2]:
                    xarea[ifstep, i] = xareasite[2]
                else:
                    xarea[ifstep, i] = xareasite[3]
                outfile1.write("%5.1f " % xarea[ifstep, i])
            outfile1.write('\n')

        # generate flow data
        flowsite = [190, 200, 225, 270]  # discharge for each site
        flow = zeros((totalfstep, nrch), float64)
        for ifstep in range(totalfstep):
            for i in range(nrch):
                if i < lstrchno[0]:
                    flow[ifstep, i] = flowsite[0]
                elif lstrchno[0] <= i < lstrchno[1]:
                    flow[ifstep, i] = flowsite[1]
                elif lstrchno[1] <= i < lstrchno[2]:
                    flow[ifstep, i] = flowsite[2]
                else:
                    flow[ifstep, i] = flowsite[3]
                outfile2.write("%3d " % flow[ifstep, i])
            outfile2.write('\n')
    else:
        input_entry = 'UNSTEADYQ.INP'
        input_name = os.path.join(path, input_entry)
        infile = open(input_name, 'r')
        file_content = infile.readlines()
        num_lines = len(file_content)
        infile.seek(0)
        # print num_lines
        qtstep, nfstep, nx, flowloc_otis, flow_otis, xarea_otis = \
            read_data_otis(num_lines, infile)
        # qtstep = flow timestep, nfstep = number of flow timesteps
        # output flows and areas
        t_hr = 11.3
        for its in range(totalfstep + 1):  # include an initial flow
            outfile1.write("%6.3f " % t_hr)
            outfile2.write("%6.3f " % t_hr)
            for idx in range(nx):
                outfile1.write(" %8.6f " % xarea_otis[its, idx])
                outfile2.write(" %8.6f " % flow_otis[its, idx])
            t_hr += qtstep
            outfile1.write('\n')
            outfile2.write('\n')
    ############################################################################
    # create arrays for flows, areas, inflows, and inflow concs used for the SWT
    ############################################################################
    # read flows and areas except the first line of them
    x_values = zeros(nx, float)
    y_values_area = zeros(nx, float)
    y_values_flow = zeros(nx, float)
    xarea = zeros((totalfstep, nrch), float64)
    flow = zeros((totalfstep, nrch), float64)
    inflow_swt = zeros((totalfstep, nusbc), float64)
    inflowconc_swt = zeros((totalfstep, nusbc), float64)
    # the following numbers are from flow model data
    qconc = 0.2958
    injstp_start = 8
    injstp_end = 233
    # calculate x values which are centers of all the cells
    x = calcx(nrch, delx)

    for i in range(totalfstep):
        j = i + 1
        for k in range(nx):
            x_values[k] = flowloc_otis[k]
            y_values_area[k] = xarea_otis[j, k]
            y_values_flow[k] = flow_otis[j, k]
            # set inflows and inflow concs
            if k == 0:
                inflow_swt[i, k] = flow_otis[j, k]
                outfile3.write("%8.6f\n" % inflow_swt[i, k])
                if injstp_start <= j <= injstp_end:
                    inflowconc_swt[i, k] = qconc / inflow_swt[i, k]
                else:
                    inflowconc_swt[i, k] = 0.0
                outfile4.write("%8.6f\n" % inflowconc_swt[i, k])
        for k in range(nrch):
            # interpolate flows and areas for those x's
            y_area_interp = scipy.interpolate.interp1d(x_values, y_values_area)
            xarea[i, k] = y_area_interp(x[k])
            y_flow_interp = scipy.interpolate.interp1d(x_values, y_values_flow)
            flow[i, k] = y_flow_interp(x[k])

    if flow_data_title == 'Monocacy River Case 5':
        # generate inflow data
        inflowvalue = 190
        for fstep in range(totalfstep):
            for i in range(nusbc):
                inflow_swt[fstep, i] = inflowvalue
                outfile3.write("%3d " % inflow_swt[fstep, i])
            outfile3.write('\n')

        # generate inflow conc data
        inflowconcvalue = 0.0
        for fstep in range(totalfstep):
            for i in range(nusbc):
                if fstep == 0:
                    inflowconc_swt[fstep, i] = iconinflow[i]  # initial inflow
                    # conc
                else:
                    inflowconc_swt[fstep, i] = inflowconcvalue
                outfile4.write("%6.2f " % inflowconc_swt[fstep, i])
            outfile4.write('\n')

    infile.close()
    outfile1.close()
    outfile2.close()
    outfile3.close()
    outfile4.close()

    return xarea, flow, inflow_swt, inflowconc_swt


def generate_flow_data(totalfstep, nrch, nusbc, iconinflow, iniconc,
                       delx, rvsfdata):
    """ This function creates cross-sectional area, flow, inflow, and inflow
    concentration data for numerical models automatically.
    It is revised from "generate_flow_model_data" to compare numerical solutions
    with analytical solutions on Jan. 2019.
    """
    # set directory for output files
    dfltsubdir = config.dfltsubdir
    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, dfltsubdir)

    # create output file
    output_entry1 = 'XArea_4review.txt'
    output_name1 = os.path.join(path, output_entry1)
    outfile1 = open(output_name1, 'w')
    output_entry2 = 'Flow_4review.txt'
    output_name2 = os.path.join(path, output_entry2)
    outfile2 = open(output_name2, 'w')
    output_entry3 = 'Inflow_4review.txt'
    output_name3 = os.path.join(path, output_entry3)
    outfile3 = open(output_name3, 'w')
    output_entry4 = 'InflowConc_4review.txt'
    output_name4 = os.path.join(path, output_entry4)
    outfile4 = open(output_name4, 'w')

    # generate cross-sectional area data----------------------------------------
    # revised for reverse flow on Jan. 2019
    xarea = zeros((totalfstep, nrch), float64)
    flow = zeros((totalfstep, nrch), float64)
    if rvsfdata[0] != 0:
        # generate cross-sectional area data
        for ifstep in range(totalfstep):
            for i in range(nrch):
                xarea[ifstep, i] = delx
                outfile1.write("%5.1f " % xarea[ifstep, i])
            outfile1.write('\n')
        # generate flow data
        # define timesteps for flow to change directions
        nrvs = rvsfdata[0]  # no. of flow reversals
        tstep_rvs = zeros(nrvs)  # flow reverse timesteps
        for i in range(nrvs):
            tstep_rvs[i] = rvsfdata[i + 1]
        # assign flow values (positive or negative)
        # depending on the direction of flow
        flowfactor = 1  # factor for changing flow direction
        irvs = 0  # flow reversal counter
        for ifstep in range(totalfstep):
            for i in range(nrch):
                flow[ifstep, i] = flowfactor * rvsfdata[nrvs + 1]
                outfile2.write("%3d " % flow[ifstep, i])
            outfile2.write('\n')
            if ifstep == (tstep_rvs[irvs] - 2):
                flowfactor *= -1
                if irvs != nrvs - 1:
                    irvs += 1

    # calculate inflow and inflow concentrations--------------------------------
    # initialize inflow & inflow concentrations
    inflow = zeros((totalfstep, nusbc), float64)
    inflowconc = zeros((totalfstep, nusbc), float64)

    # generate inflow data
    for fstep in range(totalfstep):
        for i in range(nusbc):
            inflow[fstep, i] = flow[fstep, i]
            outfile3.write("%3d " % inflow[fstep, i])
        outfile3.write('\n')

    # generate inflow conc data
    for fstep in range(totalfstep):
        for i in range(nusbc):
            if fstep == 0:
                inflowconc[fstep, i] = iconinflow[i]
                # initial inflow
                # conc
            else:
                inflowconc[fstep, i] = iniconc
            outfile4.write("%6.2f " % inflowconc[fstep, i])
        outfile4.write('\n')

    outfile1.close()
    outfile2.close()
    outfile3.close()
    outfile4.close()

    return xarea, flow, inflow, inflowconc


def generate_storage_parameters(modeloption, totalfstep, nrch, sareasite,
                                sratesite, lstrchno):
    """ This function creates storage zone area and storage rate."""
    # set directory for output files
    dfltsubdir = config.dfltsubdir
    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, dfltsubdir)

    # create output file
    if modeloption != 0:
        output_entry1 = 'StorageZoneArea.txt'
        output_name1 = os.path.join(path, output_entry1)
        outfile1 = open(output_name1, 'w')
        output_entry2 = 'StorageRate.txt'
        output_name2 = os.path.join(path, output_entry2)
        outfile2 = open(output_name2, 'w')

    # set the last reach number for each site
    # case for 114 reaches
    if nrch == 114:
        lstrchno = [29, 54, 85, 114]  # last reach number
    elif nrch == 427:
        lstrchno = [112, 204, 312, 427]  # last reach number
    # case for 4384 reaches
    elif nrch == 4384:
        lstrchno = [1127, 2105, 3215, 4384]  # last reach number
    else:
        if lstrchno is None:
            print ('Warning: Enter last reach numbers for each site')

    # generate storage zone area data
    # Monocacy River Case 5
    # sareasite = storage zone area for each site
    sarea = zeros((totalfstep, nrch), float64)
    for fstep in range(totalfstep):
        for i in range(nrch):
            if i < lstrchno[0]:
                sarea[fstep, i] = sareasite[0]
            elif lstrchno[0] <= i < lstrchno[1]:
                sarea[fstep, i] = sareasite[1]
            elif lstrchno[1] <= i < lstrchno[2]:
                sarea[fstep, i] = sareasite[2]
            else:
                sarea[fstep, i] = sareasite[3]
            if modeloption != 0:
                outfile1.write("%8.2E " % sarea[fstep, i])
        if modeloption != 0:
            outfile1.write('\n')

    # generate storage rate
    # sratesite = storage rate for each site
    srate = zeros((totalfstep, nrch), float64)
    for fstep in range(totalfstep):
        for i in range(nrch):
            if i < lstrchno[0]:
                srate[fstep, i] = sratesite[0]
            elif lstrchno[0] <= i < lstrchno[1]:
                srate[fstep, i] = sratesite[1]
            elif lstrchno[1] <= i < lstrchno[2]:
                srate[fstep, i] = sratesite[2]
            else:
                srate[fstep, i] = sratesite[3]
            if modeloption != 0:
                outfile2.write("%8.2E " % srate[fstep, i])
        if modeloption != 0:
            outfile2.write('\n')
    if modeloption != 0:
        outfile1.close()
        outfile2.close()

    return sarea, srate
