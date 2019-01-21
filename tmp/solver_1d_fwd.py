from numpy import zeros, linalg
from sparse_matrix import *
# import os

# --Surface Water Routing Transport Solver


class SWTSolver:
    """Solving ADEs using implicit finite difference methods
    for the second problem of bi-directional flow
    Because of lack of dispersion, dispersion terms
    Because of lack of dispersion, dispersion terms
    across the inlet and outlet interface are ignored.
    
    The revision from solver_1d_rev5.py is identified by "# bi-dir" 
    (for bi-directional flow)    

    Attributes:
    t: time value
    c: array of solution values (at time point t)
    dt: time step
    """

    def __init__(self, dt, em, alpha, omega, nrch, iac, iap, jap, delx,
                 area, flow, inflow, inflowconc, inflowconcn, sarea, srate,
                 dsbound, solopt, idspflg, sdecay, tmin, modeloptn, r):
        self.dt = dt
        self.em = em
        self.alpha = alpha
        self.omega = omega
        self.nrch = nrch
        self.modeloptn = modeloptn
        self.iac = iac
        self.iap = iap
        self.jap = jap
        self.delx = delx
        self.area = area
        self.flow = flow
        self.inflow = inflow
        self.inflowconc = inflowconc
        self.inflowconcn = inflowconcn
        # storage parameters
        self.sarea = sarea
        self.srate = srate
        self.sdecay = sdecay
        self.tmin = tmin
        self.r = r
        self.dsbound = dsbound
        self.solopt = solopt
        self.idspflg = idspflg
        self.a = None
        self.b = None
        self.c = None
        self.sc = None
        self.t = None
        self.iusbc = None
        self.idsbc = None
        self.k = None
        self.eadv = None
        self.hadv = None          # bi-dir
        self.edisp = None
        self.hdisp = None         # bi-dir
        self.gadv = None
        self.jadv = None          # bi-dir
        self.gdisp = None
        self.jdisp = None         # bi-dir
        self.fdt = None
        self.idt = None           # bidir
        self.fadv = None
        self.iadv = None          # bi-dir
        self.fstor = None
        self.fdisp = None
        self.idisp = None         # bi-dir
        self.flown = None
        self.afacen = None
        self.arean = None
        self.adusbc = None

    def advance(self):
        """Advance solution one time step."""
        raise NotImplementedError

    def set_initial_condition(self, c0, sc0, t0=0):
        if c0 == 0.0:
            self.c = zeros(self.nrch, float)
        else:
            self.c = zeros(self.nrch, float)
            for i in range(self.nrch):
                self.c[i] = c0
        # --set storage zone initial concentration
        if sc0 == 0.0:
            self.sc = zeros(self.nrch, float)
        # sc = storage zone conc.
        else:
            self.sc = zeros(self.nrch, float)
            for i in range(self.nrch):
                self.sc[i] = sc0
        self.t = t0
        return self.c

    def set_previous_condition(self, cprev, scprev, tprev):
        self.c = cprev
        self.sc = scprev
        self.t = tprev

    def solve(self):
        """
        Advance solution from t = t0 to t = total simulation time,
        steps of dt.
        """
        cnew, scnew = self.advance()
        # cnew = new conc., scnew = new storage conc.
        self.c = cnew
        self.sc = scnew
        self.t += self.dt
        return self.c, self.sc, self.t

    def arraysolver(self):
        a, b = self.a, self.b
        cnew = linalg.solve(a, b)
        return cnew

    # --calculate dispersion coefficient @ interface segments i, i+1
    def dface(self, j, jj=None):
        em, delx, nrch = self.em, self.delx, self.nrch
        if j != nrch - 1:
            dfacevalue = (em[j] * delx[jj] + em[jj] * delx[j]) / (
                delx[j] + delx[jj])
        else:
            dfacevalue = em[nrch - 1]
        return dfacevalue

    # --calculate cross sectional area @ interface segments i, i+1
    def aface(self, j, jj=None):
        area, delx, nrch = self.area, self.delx, self.nrch
        if j != nrch - 1:
            afacevalue = (area[j] * delx[jj] + area[jj] * delx[j]) / (
                delx[j] + delx[jj])
        else:
            afacevalue = area[nrch - 1]
        return afacevalue

    # --save variables from previous time step and preprocess
    def saven(self):
        nrch, inflow, flow, area, flown, afacen, arean, iusbc = self.nrch, \
            self.inflow, self.flow, self.area, self.flown, self.afacen, \
            self.arean, self.iusbc

        # initialize flow, interface area, and area at previous timestep
        flown = zeros(2, float)
        afacen = zeros(2, float)
        arean = zeros(nrch, float)

        # define the above values at timestep n
        # these variables at 'n' only used for the first & last segments
        flown[0] = flow[0]
        flown[1] = flow[nrch - 1]
        for i in range(nrch):
            if i == 0:
                ip1 = i + 1
                afacen[0] = self.aface(i, ip1)
            elif i == nrch - 1:
                afacen[1] = self.aface(i)
            else:
                pass

        # define arean at timestep n
        for irch in range(nrch):
            arean[irch] = area[irch]
        return flown, afacen, arean

    # --save variables from previous time step and preprocess
    #   for a scheme without advection term
    def saven_woadv(self):
        nrch, inflow, flow, area, flown, afacen, arean, iusbc = self.nrch, \
            self.inflow, self.flow, self.area, self.flown, self.afacen, \
            self.arean, self.iusbc

        # initialize interface area, and area at previous timestep
        afacen = zeros(2, float)
        arean = zeros(nrch, float)

        # define the above values at timestep n
        # these variables at 'n' only used for the first & last segments
        for i in range(nrch):
            if i == 0:
                ip1 = i + 1
                afacen[0] = self.aface(i, ip1)
            elif i == nrch - 1:
                afacen[1] = self.aface(i)
            else:
                pass

        # define arean at timestep n
        for irch in range(nrch):
            arean[irch] = area[irch]
        return afacen, arean


class Dispersion:
    # --calculate model dispersion
    # revise numerical dispersion functions below if omega is not set to 0.5
    def __init__(self, path, alpha, nrch, dispcoefopt, dcoef, i, vel, delx):
        self.path = path
        self.alpha = alpha
        self.nrch = nrch
        self.dispcoefopt = dispcoefopt
        self.dcoef = dcoef
        self.i = i
        self.vel = vel
        self.delx = delx

    def modeldisp(self):
        path, alpha, nrch, dispcoefopt, dcoef, i, vel, delx = self.path, \
            self.alpha, self.nrch, self.dispcoefopt, self.dcoef, self.i, \
            self.vel, self.delx

        if alpha != 0.0:
            en = 0.5 * vel * delx  # numerical dispersion
            if en < dcoef:
                if dispcoefopt == 1:
                    em = dcoef  # model dispersion
                else:
                    em = dcoef - en
                errormessage = 'No error!'
            else:
                em = dcoef
                errormessage = ('Warning: The numerical dispersion, %5.1f, is '
                                'bigger than the model dispersion, %5.1f,'
                                ' for delta x(%d) = %5.1f.\n' % (en, dcoef,
                                                                 i + 1, delx))
        else:
            em = dcoef
            if vel != 0.0:
                delxc = em / (0.5 * vel)
                if (delx - delxc) > 0.0:
                    errormessage = ('Warning: The delta x(%d) = %5.1f must be'
                                    ' smaller than the critical segment length'
                                    ' of %5.2f.\n' % (i + 1, delx, delxc))
                else:
                    errormessage = 'No error!'
            else:
                errormessage = 'No error!'
        return em, errormessage


class WeightedFiniteDiff(SWTSolver):
    """Solving ADEs using implicit finite difference methods
    Because of lack of dispersion, dispersion terms across the inlet and
    outlet interface are ignored.

    Attributes:
    a: coefficient matrix
    b: right hand side of ADEs
    """

    def advance(self):
        nrch, iac, iap, jap, solopt = self.nrch, self.iac, self.iap, \
                                      self.jap, self.solopt
        # --calculate a and b matrices
        self.a = zeros((nrch, nrch), float)
        self.b = zeros(nrch, float)
        self.iusbc = 0  # index for upstream boundary conditions
        self.idsbc = 0  # index for downstream boundary conditions
        for irch in range(nrch):
            self.k = zeros(iac[irch], int)
            for icc in range(iac[irch]):
                self.k[icc] = jap[iap[irch] + icc]
            if (iac[irch] == 2) and (self.k[0] < self.k[1]):
                self.a, self.b = self.firstrch(irch)
                self.iusbc += 1
            elif (iac[irch] == 2) and (self.k[0] > self.k[1]):
                self.a, self.b = self.lastrch(irch)
                self.idsbc += 1
            else:
                self.a, self.b = self.interiorch(irch)
        # print self.a
        # print self.b
        if solopt == 1:
            cnew = self.arraysolver()
        else:
            if solopt == 2:
                solver = DSSuperLU(self.a, self.b, nrch)
            else:
                solver = DsolIncomLU(self.a, self.b, nrch)
            cnew = solver.array_solver()
        scnew = self.calculate_storage_conc(cnew)
        # print cnew
        # print scnew
        return cnew, scnew

    def firstrch(self, irch):
        a, b, k = self.a, self.b, self.k
        a[irch, k[0]] = self.diagonal_amatrix(k[0], ip1=k[1])
        a[irch, k[1]] = self.upper_tria(k[0], k[1])
        # --assign default values for eadv and edisp before calling
        # right_hand_side_b function
        self.eadv = 0.0
        self.edisp = 0.0
        b[irch] = self.right_hand_side_b(k[0], ip1=k[1])
        return a, b

    def interiorch(self, irch):
        a, b, k = self.a, self.b, self.k
        uscnt = 0  # count upstream reaches
        dscnt = 0  # count downstream reaches
        diagasum = 0.0
        bsum = 0.0
        # --count upstream and downstream reaches
        for i in range(1, len(k)):
            if k[i] < k[0]:
                uscnt += 1
            else:
                dscnt += 1
        uscntp1 = uscnt + 1
        dscntp1 = dscnt + 1
        fcnt = 0  # if l = 0, calculate fdt
        # --calculate a and b matrices
        for i in range(1, uscntp1):
            a[irch, k[i]] = self.lower_tria(k[0], k[i])
            bsgl = self.right_hand_side_b(k[0], im1=k[i])
            bsum += bsgl
        for i in range(1, dscntp1):
            idc = i + uscnt
            a[irch, k[idc]] = self.upper_tria(k[0], k[idc])
            bsgl = self.right_hand_side_b(k[0], ip1=k[idc])
            bsum += bsgl
        for i in range(1, uscntp1):
            diagasgl = self.diagonal_amatrix(k[0], l=fcnt, im1=k[i])
            bsgl = self.right_hand_side_b(k[0], l=fcnt)
            diagasum += diagasgl
            bsum += bsgl
            fcnt += 1
        for i in range(1, dscntp1):
            idc = i + uscnt
            diagasgl = self.diagonal_amatrix(k[0], l=fcnt, ip1=k[idc])
            bsgl = self.right_hand_side_b(k[0], l=fcnt)
            diagasum += diagasgl
            bsum += bsgl
            fcnt += 1
        a[irch, k[0]] = diagasum
        b[irch] = bsum
        return a, b

    def lastrch(self, irch):
        a, b, k = self.a, self.b, self.k
        a[irch, k[1]] = self.lower_tria(k[0], k[1])
        a[irch, k[0]] = self.diagonal_amatrix(k[0], im1=k[1])
        # --assign default values for gadv and gdisp before calling
        # right_hand_side_b function
        self.gadv = 0.0
        self.gdisp = 0.0
        b[irch] = self.right_hand_side_b(k[0], im1=k[1])
        return a, b

    # --calculate lower triangular part of matrix a
    def lower_tria(self, i, im1):
        omega, flow, delx, alpha, dt, idspflg = self.omega, self.flow, \
            self.delx, self.alpha, self.dt, self.idspflg

        self.eadv = -flow[i] / (delx[im1] + delx[i]) * (
            alpha * delx[im1] + delx[i])
        eadvw = omega * self.eadv
        # --the dispersion flag is used to avoid calculating dipsersion terms
        # between diversion ditch and its upstream reach.
        if idspflg[i] != 0:
            adm1 = self.aface(im1, i) * self.dface(im1, i)
            self.edisp = -2.0 * adm1 / (delx[im1] + delx[i])
            edispw = omega * self.edisp
        else:
            self.edisp = 0.0
            edispw = omega * self.edisp
        avalue = dt * (eadvw + edispw)
        return avalue

    # --calculate diagonal part of matrix a
    def diagonal_amatrix(self, i, l=0, im1=None, ip1=None):
        area, omega, flow, delx, em, alpha, dt, iac, idspflg, srate, sarea, \
            sdecay, t, tmin, r, adusbc = self.area, self.omega, \
            self.flow, self.delx, self.em, self.alpha, self.dt, self.iac, \
            self.idspflg, self.srate, self.sarea, self.sdecay, self.t, \
            self.tmin, self.r, self.adusbc
        if l == 0:
            self.fdt = area[i] * delx[i] / dt
            # self.fdt = 0.0
        else:
            self.fdt = 0.0

        # use calculate_storage_parameters function
        epsilon, gmm = self.calculate_storage_parameters(i)
        # epsilonr is only for SD model
        epsilonr = epsilon * r
        # gmm = srate[i] * dt * area[i] / sarea[i]  # gmm = gamma

        # make storage term simpler
        oneplus = 1 + omega * sdecay * dt
        twoplus = 1 + omega * gmm + omega * dt * sdecay

        # first segment
        if (iac[i] == 2) and im1 is None:
            self.fadv = flow[i] / (delx[i] + delx[ip1]) * \
                (alpha * delx[i] + delx[ip1])
            adp1 = self.aface(i, ip1) * self.dface(i, ip1)
            adusbc = adp1
            self.fdisp = 2.0 * (adusbc / delx[i] + adp1 / (
                delx[i] + delx[ip1]))
            fadvw = omega * self.fadv
            fdispw = omega * self.fdisp
            if self.modeloptn == 0:
                self.fstor = 0.0
                fstorw =  omega * self.fstor
            elif self.modeloptn == 1:
                self.fstor = srate[i] * area[i] * delx[i] * oneplus / twoplus
                fstorw = omega * self.fstor
            else:
                self.fstor = epsilonr * area[i] * delx[i] * oneplus / twoplus
                fstorw = omega * self.fstor
        # last segment
        elif (iac[i] == 2) and ip1 is None:
            self.fadv = -flow[i] * ((1 - alpha) * delx[im1] / (delx[im1] +
                                                               delx[i]) - 1)
            # self.fadv = -flow[i]
            fadvw = omega * self.fadv
            if self.modeloptn == 0:
                self.fstor = 0.0
                fstorw = omega * self.fstor
            elif self.modeloptn == 1:
                self.fstor = srate[i] * area[i] * delx[i] * oneplus / twoplus
                fstorw = omega * self.fstor
            else:
                self.fstor = epsilonr * area[i] * delx[i] * oneplus / twoplus
                fstorw = omega * self.fstor
            if idspflg[i] != 0:
                adm1 = self.aface(im1, i) * self.dface(im1, i)
                self.fdisp = (2 * adm1 / (delx[im1] + delx[i]))
                fdispw = omega * self.fdisp
            else:
                self.fdisp = 0.0
                fdispw = omega * self.fdisp
        # interior segments
        else:
            # storage
            if l == 0:
                if self.modeloptn == 0:
                    self.fstor = 0.0
                    fstorw = omega * self.fstor
                elif self.modeloptn == 1:
                    self.fstor = srate[i] * area[i] * delx[i] * \
                        oneplus / twoplus
                    fstorw = omega * self.fstor
                else:
                    self.fstor = epsilonr * area[i] * delx[i] * \
                        oneplus / twoplus
                    fstorw = omega * self.fstor
            else:
                fstorw = 0.0
            # adv and disp
            if im1 is not None:
                self.fadv = -(flow[im1] / (delx[im1] + delx[i]) *
                             (1 - alpha) * delx[im1])
                # self.fadv = 0.0
                fadvw = omega * self.fadv
                if idspflg[i] != 0:
                    adm1 = self.aface(im1, i) * self.dface(im1, i)
                    self.fdisp = 2.0 * (adm1 / (delx[im1] + delx[i]))
                    fdispw = omega * self.fdisp
                else:
                    self.fdisp = 0.0
                    fdispw = omega * self.fdisp
            else:
                self.fadv = flow[i] / (delx[i] + delx[ip1]) * \
                    (alpha * delx[i] + delx[ip1])
                fadvw = omega * self.fadv
                if idspflg[ip1] != 0:
                    adp1 = self.aface(i, ip1) * self.dface(i, ip1)
                    self.fdisp = 2.0 * (adp1 / (delx[i] + delx[ip1]))
                    fdispw = omega * self.fdisp
                else:
                    self.fdisp = 0.0
                    fdispw = omega * self.fdisp
        avalue = dt * (self.fdt + fadvw + fdispw + fstorw)
        return avalue

    # --calculate upper triangular part of matrix a
    def upper_tria(self, i, ip1):
        omega, flow, delx, alpha, dt, idspflg = self.omega, self.flow, \
            self.delx, self.alpha, self.dt, self.idspflg

        self.gadv = flow[i] / (delx[i] + delx[ip1]) * (1 - alpha) * delx[i]
        gadvw = omega * self.gadv
        if idspflg[ip1] != 0:
            adp1 = self.aface(i, ip1) * self.dface(i, ip1)
            self.gdisp = -2.0 * adp1 / (delx[i] + delx[ip1])
            gdispw = omega * self.gdisp
        else:
            self.gdisp = 0.0
            gdispw = omega * self.gdisp
        avalue = dt * (gadvw + gdispw)
        return avalue

    # --calculate right hand side
    def right_hand_side_b(self, i, l=0, im1=None, ip1=None):
        fdt, c, omega, fadv, gadv, inflow, inflowconc, fdisp, gdisp, em, \
            area, delx, dt, eadv, flow, alpha, dsbound, edisp, iusbc, idsbc, \
            iac, srate, sarea, sdecay, sc, t, tmin, r, fstor, inflowconcn, \
            nrch, flown, afacen, arean = self.fdt, self.c, self.omega, \
            self.fadv, self.gadv, self.inflow, self.inflowconc, self.fdisp, \
            self.gdisp, self.em, self.area, self.delx, self.dt, self.eadv, \
            self.flow, self.alpha, self.dsbound, self.edisp, self.iusbc, \
            self.idsbc, self.iac, self.srate, self.sarea, self.sdecay, \
            self.sc, self.t, self.tmin, self.r, self.fstor, self.inflowconcn, \
            self.nrch, self.flown, self.afacen, self.arean

        # use calculate_storage_parameters function
        if self.modeloptn == 0:
            # epsilon and gamma are not used for the ADE
            epsilon = 0.0
            gmm = 0.0
        elif self.modeloptn == 1:
            epsilon, gmm = self.calculate_storage_parameters(i)
            # epsilon is none.
        else:
            epsilon, gmm = self.calculate_storage_parameters(i)

        epsilonr = epsilon * r
        # gmm = srate[i] * dt * area[i] / sarea[i]  # gmm = gamma

        # make storage term simpler
        twoplus = 1 + omega * gmm + omega * dt * sdecay

        if i == 0 or i == nrch - 1:
            flown, afacen, arean = self.saven()

        # first segment
        if (iac[i] == 2) and im1 is None:
            rdt = fdt * c[i]
            radv = -(1 - omega) * fadv * c[i] - (1 - omega) * gadv * c[ip1] + \
                (1 - omega) * flown[0] * inflowconcn[iusbc] + omega * \
                inflow[iusbc] * inflowconc[iusbc]
            rdisp = -(1 - omega) * fdisp * c[i] - (1 - omega) * gdisp * \
                c[ip1] + 2.0 * self.dface(i, ip1) / delx[i] * (
                afacen[0] * (1 - omega) * inflowconcn[iusbc] +
                self.aface(i, ip1) * omega * inflowconc[iusbc])
            # assume A_01 = A_12 and D_01= D_12

            if self.modeloptn == 0:
                rstor = 0.0
            elif self.modeloptn == 1:
                rstor = -fstor * (1 - omega) * c[i] + srate[i] * area[i] * \
                    delx[i] / twoplus * sc[i]
            else:
                rstor = -fstor * (1 - omega) * c[i] + epsilonr * area[i] * \
                    delx[i] / twoplus * sc[i]
            inflowconcn[iusbc] = inflowconc[iusbc]
            bvalue = dt * (rdt + radv + rdisp + rstor)

        # last segment
        elif (iac[i] == 2) and ip1 is None:
            rdt = fdt * c[i]
            # adv
            radv = -(1 - omega) * eadv * c[im1] - (1 - omega) * fadv * c[i] - \
                 (1 - alpha) * delx[i] * dsbound[idsbc] / (2.0 * em[i]) * (
                     flown[1] * (1 - omega) + flow[i] * omega)
            # disp
            rdisp = -(1 - omega) * edisp * c[im1] - (1 - omega) * fdisp * \
                c[i] + dsbound[idsbc] * ((1 - omega) * afacen[1] + omega *
                                         self.aface(i))
            if self.modeloptn == 0:
                rstor = 0.0
            elif self.modeloptn == 1:
                rstor = -fstor * (1 - omega) * c[i] + srate[i] * area[i] * \
                    delx[i] / twoplus * sc[i]
            else:
                rstor = -fstor * (1 - omega) * c[i] + epsilonr * area[i] * \
                    delx[i] / twoplus * sc[i]
            bvalue = dt * (rdt + radv + rdisp + rstor)

        # interior segments
        else:
            # adv & disp
            if im1 is not None:
                radv = -(1 - omega) * eadv * c[im1]
                rdisp = -(1 - omega) * edisp * c[im1]
                bvalue = dt * (radv + rdisp)
            elif ip1 is not None:
                radv = -(1 - omega) * gadv * c[ip1]
                rdisp = -(1 - omega) * gdisp * c[ip1]
                bvalue = dt * (radv + rdisp)
            else:
                # storage
                if l == 0:
                    if self.modeloptn == 0:
                        rstor = 0.0
                    elif self.modeloptn == 1:
                        rstor = -fstor * (1 - omega) * c[i] + srate[i] \
                            * area[i] * delx[i] / twoplus * sc[i]
                    else:
                        rstor = -fstor * (1 - omega) * c[i] + epsilonr \
                            * area[i] * delx[i] / twoplus * sc[i]
                else:
                    rstor = 0.0
                rdt = fdt * c[i]
                radv = -(1 - omega) * fadv * c[i]
                rdisp = -(1 - omega) * fdisp * c[i]
                bvalue = dt * (rdt + radv + rdisp + rstor)
        return bvalue

    # --calculate storage concs
    def calculate_storage_conc(self, cnew):
        nrch, omega, dt, sdecay, c, sc = self.nrch, \
            self.omega, self.dt, self.sdecay, self.c, self.sc
        # Initialize the new storage concentration.
        scnew = zeros(nrch, float)
        for i in range(nrch):
            # use calculate_storage_parameters function
            if self.modeloptn == 0:
                # epsilon and gamma are not used for the ADE
                epsilon = 0.0
                gmm = 0.0
            elif self.modeloptn == 1:
                epsilon, gmm = self.calculate_storage_parameters(i)
                # epsilon is none.
            else:
                epsilon, gmm = self.calculate_storage_parameters(i)
            if self.modeloptn == 0:
                scnew[i] = 0.0
            else:
                scnew[i] = ((1 - (1 - omega) * gmm - (
                    1 - omega) * dt * sdecay) * sc[i] + gmm * (
                    omega * cnew[i] + (1 - omega) * c[i])) / (
                    1 + omega * gmm + omega * dt * sdecay)
        return scnew

    def calculate_storage_parameters(self, i):
        # define gamma for storage term
        srate, dt, area, sarea, t, tmin = self.srate, self.dt, self.area, \
            self.sarea, self.t, self.tmin
        # calculate epsilon and gamma
        if self.modeloptn == 0:
            # epsilon and gamma are not used for the ADE
            epsilon = 0.0
            gmm = 0.0
        elif self.modeloptn == 1:
            epsilon = 0.0  # epsilon is not used for TS model
            gmm = srate[i] * dt * area[i] / sarea[i]  # gmm = gamma
        else:
            if t <= tmin:
                epsilon = 1 / tmin
            else:
                epsilon = 1 / t
            gmm = epsilon * dt
        return epsilon, gmm


class CrankNicolson(WeightedFiniteDiff):
    """ SWT model verification using Crank Nicolson Scheme from OTIS by
    Robert L. Runkel
    Test problem-Example 12.1 and Synthetic example 2
    """
    def advance(self):
        nrch, solopt, dt = self.nrch, self.solopt, self.dt
        # --calculate a and b matrices
        self.a = zeros((nrch, nrch), float)
        self.b = zeros(nrch, float)
        self.iusbc = 0
        self.idsbc = 0
        for irch in range(nrch):
            if irch == 0:
                self.a, self.b = self.firstrch(irch)
                self.iusbc += 1
            elif irch == (nrch - 1):
                self.a, self.b = self.lastrch(irch)
                self.idsbc += 1
            else:
                self.a, self.b = self.interiorch(irch)
        # print self.a
        # print self.b
        if solopt == 1:
            cnew = self.arraysolver()
        else:
            if solopt == 2:
                solver = DSSuperLU(self.a, self.b, nrch)
            else:
                solver = DsolIncomLU(self.a, self.b, nrch)
            cnew = solver.array_solver()
        scnew = self.calculate_storage_conc(cnew)
        # print cnew
        # print scnew
        return cnew, scnew

    def firstrch(self, irch):
        a, b = self.a, self.b
        irchp1 = irch + 1
        a[irch, irch] = self.diagonal_amatrix(irch, ip1=irchp1)
        a[irch, irchp1] = self.upper_tria(irch, irchp1)
        # --assign default values for eadv and edisp before calling
        # right_hand_side_b function
        self.eadv = 0.0
        self.edisp = 0.0
        b[irch] = self.right_hand_side_b(irch, ip1=irchp1)
        return a, b

    def interiorch(self, irch):
        a, b = self.a, self.b
        # --calculate a and b matrices
        irchm1 = irch - 1
        irchp1 = irch + 1
        a[irch, irchm1] = self.lower_tria(irch, irchm1)
        a[irch, irchp1] = self.upper_tria(irch, irchp1)
        a[irch, irch] = self.diagonal_amatrix(irch, im1=irchm1, ip1=irchp1)
        b[irch] = self.right_hand_side_b(irch, im1=irchm1, ip1=irchp1)
        return a, b

    def lastrch(self, irch):
        a, b = self.a, self.b
        irchm1 = irch - 1
        a[irch, irchm1] = self.lower_tria(irch, irchm1)
        a[irch, irch] = self.diagonal_amatrix(irch, im1=irchm1)
        # --assign default values for gadv and gdisp before calling
        # right_hand_side_b function
        self.gadv = 0.0
        self.gdisp = 0.0
        b[irch] = self.right_hand_side_b(irch, im1=irchm1)
        return a, b

    # --calculate lower triangular part of matrix a
    def lower_tria(self, i, im1):
        flow, delx, dt, idspflg, area = self.flow, self.delx, \
            self.dt, self.idspflg, self.area

        self.eadv = -flow[i] / (2.0 * area[i] * (
            delx[im1] + delx[i]))
        # --the dispersion flag is used to avoid calculating dipsersion terms
        # between diversion ditch and its upstream reach.
        if idspflg[i] != 0:
            adm1 = self.aface(im1, i) * self.dface(im1, i)
            self.edisp = - adm1 / (area[i] * delx[i] * (delx[im1] + delx[i]))
        else:
            self.edisp = 0.0
        avalue = dt * (self.eadv + self.edisp)
        return avalue

    # --calculate diagonal part of matrix a
    def diagonal_amatrix(self, i, l=0, im1=None, ip1=None):
        area, flow, delx, dt, idspflg, srate, sdecay, r, em, adusbc = \
            self.area, self.flow, self.delx, self.dt, self.idspflg, \
            self.srate, self.sdecay, self.r, self.em, self.adusbc

        if l == 0:
            self.fdt = 1. / dt
        else:
            self.fdt = 0.0

        # use calculate_storage_parameters function
        epsilon, gmm = self.calculate_storage_parameters(i)
        # epsilonr is only for SD model
        epsilonr = epsilon * r
        # gmm = srate[i] * dt * area[i] / sarea[i]  # gmm = gamma

        if im1 is None:             # first segment
            self.fadv = flow[i] * delx[ip1] / (2.0 * area[i] * delx[i] *
                                               (delx[i] + delx[ip1]))
            if idspflg[i] != 0:
                adp1 = self.aface(i, ip1) * self.dface(i, ip1)
                adusbc = adp1
                self.fdisp = 1.0 / (area[i] * delx[i]) \
                    * (adp1 / (delx[i] + delx[ip1]) + adusbc / delx[i])
                # The above line is to include dispersion term for the entry
                # interface
            else:
                self.fdisp = 0.0

            if self.modeloptn == 0:
                self.fstor = 0.0
            elif self.modeloptn == 1:
                self.fstor = srate[i] / 2.0 * (1.0 - gmm / (2.0 + gmm + dt *
                                               sdecay))
            else:
                self.fstor = epsilonr / 2.0 * (1.0 - gmm / (2.0 + gmm + dt *
                                               sdecay))
        elif ip1 is None:           # last segment
            self.fadv = flow[i] / (2.0 * area[i] * delx[i]) * \
                (1.0 - delx[im1] / (delx[im1] + delx[i]))
            # self.fadv = -flow[i]
            if idspflg[i] != 0:
                adm1 = self.aface(im1, i) * self.dface(im1, i)
                self.fdisp = adm1 / (area[i] * delx[i] * (delx[im1] + delx[i]))
            else:
                self.fdisp = 0.0

            if self.modeloptn == 0:
                self.fstor = 0.0
            elif self.modeloptn == 1:
                self.fstor = srate[i] / 2.0 * (1.0 - gmm / (2.0 + gmm + dt *
                                               sdecay))
            else:
                self.fstor = epsilonr / 2.0 * (1.0 - gmm / (2.0 + gmm + dt *
                                               sdecay))

        else:           # interior segment
            self.fadv = flow[i] / (2.0 * area[i] * delx[i]) * (delx[ip1] / (
                delx[ip1] + delx[i]) - delx[im1] / (delx[i] + delx[im1]))
            # self.fadv = 0.0
            if idspflg[i] != 0:
                adm1 = self.aface(im1, i) * self.dface(im1, i)
                adp1 = self.aface(i, ip1) * self.dface(i, ip1)
                self.fdisp = 1.0 / (area[i] * delx[i]) * \
                    (adm1 / (delx[im1] + delx[i]) + adp1 / (
                        delx[i] + delx[ip1]))
            else:
                self.fdisp = 0.0

            if self.modeloptn == 0:
                self.fstor = 0.0
            elif self.modeloptn == 1:
                self.fstor = srate[i] / 2.0 * (1.0 - gmm / (2.0 + gmm + dt *
                                               sdecay))
            else:
                self.fstor = epsilonr / 2.0 * (1.0 - gmm / (2.0 + gmm + dt *
                                               sdecay))

        avalue = dt * (self.fdt + self.fadv + self.fdisp + self.fstor)
        return avalue

    # --calculate upper triangular part of matrix a
    def upper_tria(self, i, ip1):
        flow, delx, dt, idspflg, area = self.flow, self.delx, self.dt, \
            self.idspflg, self.area

        self.gadv = flow[i] / (2.0 * area[i] * (delx[i] + delx[ip1]))
        if idspflg[ip1] != 0:
            adp1 = self.aface(i, ip1) * self.dface(i, ip1)
            self.gdisp = -adp1 / (area[i] * delx[i] * (delx[i] + delx[ip1]))
        else:
            self.gdisp = 0.0
        avalue = dt * (self.gadv + self.gdisp)
        return avalue

    # --calculate right hand side
    def right_hand_side_b(self, i, im1=None, ip1=None):
        fdt, c, fadv, gadv, inflow, inflowconc, fdisp, gdisp, iusbc, idsbc, \
            em, area, delx, dt, eadv, flow, dsbound, edisp, srate, sdecay, \
            sc, fstor, r, inflowconcn, nrch, flown, afacen, arean = self.fdt, \
            self.c, self.fadv, self.gadv, self.inflow, self.inflowconc, \
            self.fdisp, self.gdisp, self.iusbc, self.idsbc, self.em,  \
            self.area, self.delx, self.dt, self.eadv, self.flow, self.dsbound,\
            self.edisp, self.srate, self.sdecay, self.sc, self.fstor, self.r, \
            self.inflowconcn, self.nrch, self.flown, self.afacen, self.arean

        # use calculate_storage_parameters function
        if self.modeloptn == 0:
            # epsilon and gamma are not used for the ADE
            epsilon = 0.0
            gmm = 0.0
        elif self.modeloptn == 1:
            epsilon, gmm = self.calculate_storage_parameters(i)
            # epsilon is none.
        else:
            epsilon, gmm = self.calculate_storage_parameters(i)

        epsilonr = epsilon * r
        # gmm = srate[i] * dt * area[i] / sarea[i]  # gmm = gamma

        if i == 0 or i == nrch - 1:
            flown, afacen, arean = self.saven()
        if im1 is None:   # first segment
            rdt = fdt * c[i]
            # adv
            radv = -fadv * c[i] - gadv * c[ip1] + 0.5 * (
                flown[0] * inflowconcn[iusbc] / (arean[i] * delx[i]) +
                inflow[iusbc] * inflowconc[iusbc] / (area[i] * delx[i]))
            # disp
            rdisp = -fdisp * c[i] - gdisp * c[ip1] + self.dface(i, ip1) * (
                afacen[0] * inflowconcn[iusbc] / arean[i] + self.aface(i, ip1)
                * inflowconc[iusbc] / area[i]) / (delx[i] * delx[i])
            # The above two lines are to include the dispersion term for the
            # entry interface
            if self.modeloptn == 0:
                    rstor = 0.0
            elif self.modeloptn == 1:
                rstor = -fstor * c[i] + srate[i] / 2.0 * \
                    (1.0 + (2.0 - gmm - dt * sdecay) /
                     (2.0 + gmm + dt * sdecay)) * sc[i]
            else:
                rstor = -fstor * c[i] + epsilonr / 2.0 * \
                    (1.0 + (2.0 - gmm - dt * sdecay) /
                     (2.0 + gmm + dt * sdecay)) * sc[i]
            inflowconcn[iusbc] = inflowconc[iusbc]
            bvalue = dt * (rdt + radv + rdisp + rstor)
        elif ip1 is None:    # last segment
            rdt = fdt * c[i]
            radv = -eadv * c[im1] - fadv * c[i] - 0.25 * (
                dsbound[idsbc] / self.dface(i)) * (
                flown[1] / arean[i] + flow[i] / area[i])
            rdisp = -edisp * c[im1] - fdisp * c[i] \
                + 0.5 * dsbound[idsbc] / delx[i] * (
                self.aface(i) / area[i] + afacen[1] / arean[i])
            if self.modeloptn == 0:
                    rstor = 0.0
            elif self.modeloptn == 1:
                rstor = -fstor * c[i] + srate[i] / 2.0 * \
                    (1.0 + (2.0 - gmm - dt * sdecay) /
                     (2.0 + gmm + dt * sdecay)) * sc[i]
            else:
                rstor = -fstor * c[i] + epsilonr / 2.0 * \
                    (1.0 + (2.0 - gmm - dt * sdecay) /
                     (2.0 + gmm + dt * sdecay)) * sc[i]
            bvalue = dt * (rdt + radv + rdisp + rstor)
        else:             # interior segment
            rdt = fdt * c[i]
            radv = -eadv * c[im1] - fadv * c[i] - gadv * c[ip1]
            rdisp = -edisp * c[im1] - fdisp * c[i] - gdisp * c[ip1]
            if self.modeloptn == 0:
                rstor = 0.0
            elif self.modeloptn == 1:
                rstor = -fstor * c[i] + srate[i] / 2.0 * \
                    (1.0 + (2.0 - gmm - dt * sdecay) /
                     (2.0 + gmm + dt * sdecay)) * sc[i]
            else:
                rstor = -fstor * c[i] + epsilonr / 2.0 * \
                    (1.0 + (2.0 - gmm - dt * sdecay) /
                     (2.0 + gmm + dt * sdecay)) * sc[i]
            bvalue = dt * (rdt + radv + rdisp + rstor)
        return bvalue

    # --calculate storage concs
    def calculate_storage_conc(self, cnew):
        nrch, dt, sdecay, c, sc = self.nrch, self.dt, self.sdecay, \
            self.c, self.sc
        # Initialize the new storage concentration.
        scnew = zeros(nrch, float)
        for i in range(nrch):
            # use calculate_storage_parameters function
            if self.modeloptn == 0:
                # epsilon and gamma are not used for the ADE
                epsilon = 0.0
                gmm = 0.0
            elif self.modeloptn == 1:
                epsilon, gmm = self.calculate_storage_parameters(i)
                # epsilon is none.
            else:
                epsilon, gmm = self.calculate_storage_parameters(i)
            if self.modeloptn == 0:
                scnew[i] = 0.0
            else:
                scnew[i] = ((2.0 - gmm - dt * sdecay) * sc[i] + gmm
                            * (c[i] + cnew[i])) / (2.0 + gmm + dt * sdecay)
        return scnew
