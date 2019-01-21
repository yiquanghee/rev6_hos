from numpy import zeros, linalg

from solver_1d_fwd import *


class WeightedFiniteDiffRvs(SWTSolver):
    """Solving ADEs using implicit finite difference methods 
    for the second problem of bi-directional flow
    Because of lack of dispersion, dispersion terms 
    across the inlet and outlet interface are ignored.
    
    The revision from solver_1d_rev5.py is identified by "# bi-dir" 
    (for bi-directional flow)    

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
        self.hadv = 0.0  # bi-dir
        self.hdisp = 0.0  # bi-dir
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
        icnt = 0  # if l = 0, calculate idt                 # bi-dir
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
            diagasgl = self.diagonal_amatrix(k[0], l=icnt, im1=k[i])  # bi-dir
            bsgl = self.right_hand_side_b(k[0], l=icnt)  # bi-dir
            diagasum += diagasgl
            bsum += bsgl
            icnt += 1  # bi-dir
        for i in range(1, dscntp1):
            idc = i + uscnt
            diagasgl = self.diagonal_amatrix(k[0], l=icnt, ip1=k[idc])  # bi-dir
            bsgl = self.right_hand_side_b(k[0], l=icnt)  # bi-dir
            diagasum += diagasgl
            bsum += bsgl
            icnt += 1  # bi-dir
        a[irch, k[0]] = diagasum
        b[irch] = bsum
        return a, b

    def lastrch(self, irch):
        a, b, k = self.a, self.b, self.k
        a[irch, k[1]] = self.lower_tria(k[0], k[1])
        a[irch, k[0]] = self.diagonal_amatrix(k[0], im1=k[1])
        # --assign default values for gadv and gdisp before calling
        # right_hand_side_b function
        self.jadv = 0.0  # bi-dir
        self.jdisp = 0.0  # bi-dir
        b[irch] = self.right_hand_side_b(k[0], im1=k[1])
        return a, b

    # --calculate lower triangular part of matrix a
    def lower_tria(self, i, im1):
        omega, flow, delx, alpha, dt, idspflg = self.omega, self.flow, \
                                                self.delx, self.alpha, self.dt, self.idspflg

        self.hadv = -flow[i] / (delx[im1] + delx[i]) * (  # bi-dir
            alpha * delx[im1] + delx[i])
        hadvw = omega * self.hadv  # bi-dir
        # --the dispersion flag is used to avoid calculating dipsersion terms
        # between diversion ditch and its upstream reach.
        if idspflg[i] != 0:
            adm1 = self.aface(im1, i) * self.dface(im1, i)
            self.hdisp = -2.0 * adm1 / (delx[im1] + delx[i])  # bi-dir
            hdispw = omega * self.hdisp  # bi-dir
        else:
            self.hdisp = 0.0  # bi-dir
            hdispw = omega * self.hdisp  # bi-dir
        avalue = dt * (hadvw + hdispw)  # bi-dir
        return avalue

    # --calculate diagonal part of matrix a
    def diagonal_amatrix(self, i, l=0, im1=None, ip1=None):
        [area, omega, flow, delx, em, alpha, dt, iac, idspflg, srate, sarea,
         sdecay, t, tmin, r, adusbc] = [self.area, self.omega, self.flow, self.delx, self.em, self.alpha,
                                        self.dt, self.iac, self.idspflg, self.srate, self.sarea, self.sdecay,
                                        self.t, self.tmin, self.r, self.adusbc]
        if l == 0:
            self.idt = area[i] * delx[i] / dt  # bi-dir
        else:
            self.idt = 0.0  # bi-dir

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
            self.iadv = flow[i] / (delx[i] + delx[ip1]) * (  # bi-dir
                alpha * delx[i] + delx[ip1])
            adp1 = self.aface(i, ip1) * self.dface(i, ip1)
            adusbc = adp1
            self.idisp = 2.0 * (adusbc / delx[i] + adp1 / (  # bi-dir
                delx[i] + delx[ip1]))
            iadvw = omega * self.iadv
            idispw = omega * self.idisp
            # The following code should be revised to consider transient storage
            if self.modeloptn == 0:
                self.fstor = 0.0
                fstorw = omega * self.fstor
            elif self.modeloptn == 1:
                self.fstor = srate[i] * area[i] * delx[i] * oneplus / twoplus
                fstorw = omega * self.fstor
            else:
                self.fstor = epsilonr * area[i] * delx[i] * oneplus / twoplus
                fstorw = omega * self.fstor
        # last segment
        elif (iac[i] == 2) and ip1 is None:
            self.iadv = -flow[i] * ((1 - alpha) * delx[im1] / (delx[im1] +  # bi-dir
                                                               delx[i]) - 1)
            iadvw = omega * self.iadv  # bi-dir
            # The following code should be revised to consider transient storage
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
                self.idisp = (2 * adm1 / (delx[im1] + delx[i]))  # bi-dir
                idispw = omega * self.idisp  # bi-dir
            else:
                self.idisp = 0.0  # bi-dir
                idispw = omega * self.idisp  # bi-dir
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
                self.iadv = -(flow[im1] / (delx[im1] + delx[i]) *  # bi-dir
                              (1 - alpha) * delx[im1])
                iadvw = omega * self.iadv  # bi-dir
                if idspflg[i] != 0:
                    adm1 = self.aface(im1, i) * self.dface(im1, i)
                    self.idisp = 2.0 * (adm1 / (delx[im1] + delx[i]))  # bi-dir
                    idispw = omega * self.idisp  # bi-dir
                else:
                    self.idisp = 0.0  # bi-dir
                    idispw = omega * self.idisp  # bi-dir
            else:
                self.iadv = flow[i] / (delx[i] + delx[ip1]) * (  # bi-dir
                    alpha * delx[i] + delx[ip1])
                iadvw = omega * self.iadv  # bi-dir
                if idspflg[ip1] != 0:
                    adp1 = self.aface(i, ip1) * self.dface(i, ip1)
                    self.idisp = 2.0 * (adp1 / (delx[i] + delx[ip1]))  # bi-dir
                    idispw = omega * self.idisp  # bi-dir
                else:
                    self.idisp = 0.0  # bi-dir
                    idispw = omega * self.idisp  # bi-dir
        avalue = dt * (self.idt + iadvw + idispw + fstorw)  # bi-dir
        return avalue

    # --calculate upper triangular part of matrix a
    def upper_tria(self, i, ip1):
        omega, flow, delx, alpha, dt, idspflg = self.omega, self.flow, \
                                                self.delx, self.alpha, self.dt, self.idspflg

        self.jadv = flow[i] / (delx[i] + delx[ip1]) * (1 - alpha) * delx[i]  # bi-dir
        jadvw = omega * self.jadv  # bi-dir
        if idspflg[ip1] != 0:
            adp1 = self.aface(i, ip1) * self.dface(i, ip1)
            self.jdisp = -2.0 * adp1 / (delx[i] + delx[ip1])  # bi-dir
            jdispw = omega * self.jdisp  # bi-dir
        else:
            self.jdisp = 0.0  # bi-dir
            jdispw = omega * self.jdisp  # bi-dir
        avalue = dt * (jadvw + jdispw)  # bi-dir
        return avalue

    # --calculate right hand side
    def right_hand_side_b(self, i, l=0, im1=None, ip1=None):
        """" The revised variables for bi-directional flow calculations are idt, iadv, jadv
             idisp, jdisp, hadv, and hdisp. 
             The transient storage variable should be updated 
             to simulate transient storage in this model.
        """
        [idt, c, omega, iadv, jadv, inflow, inflowconc, idisp, jdisp,
         em, area, delx, dt, hadv, flow, alpha, dsbound, hdisp, iusbc, idsbc,
         iac, srate, sarea, sdecay, sc, t, tmin, r, fstor, inflowconcn,
         nrch, flown, afacen, arean] = [self.idt, self.c, self.omega, self.iadv, self.jadv,
                                        self.inflow, self.inflowconc, self.idisp,
                                        self.jdisp, self.em, self.area, self.delx, self.dt, self.hadv,
                                        self.flow, self.alpha, self.dsbound, self.hdisp, self.iusbc,
                                        self.idsbc, self.iac, self.srate, self.sarea, self.sdecay,
                                        self.sc, self.t, self.tmin, self.r, self.fstor, self.inflowconcn,
                                        self.nrch, self.flown, self.afacen, self.arean]

        # Revise to simulate transient storage properly
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
            sdt = idt * c[i]  # bi-dir below
            sadv = -(1 - omega) * iadv * c[i] - (1 - omega) * jadv * c[ip1] + (
                (1 - omega) * flown[0] * inflowconcn[iusbc] + omega * inflow[iusbc] * inflowconc[iusbc])
            #
            sdisp = -(1 - omega) * idisp * c[i] - ((1 - omega) * jdisp *
                                                   c[ip1]) + 2.0 * self.dface(i, ip1) / delx[i] * (
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
            bvalue = dt * (sdt + sadv + sdisp + rstor)  # bi-dir

        # last segment
        elif (iac[i] == 2) and ip1 is None:
            sdt = idt * c[i]  # bi-dir below
            # adv
            sadv = -(1 - omega) * hadv * c[im1] - (1 - omega) * iadv * c[i] - (
                (1 - alpha) * delx[i] * dsbound[idsbc] / (2.0 * em[i]) * (
                    flown[1] * (1 - omega) + flow[i] * omega))
            # disp
            sdisp = -(1 - omega) * hdisp * c[im1] - (1 - omega) * idisp * c[i] + dsbound[idsbc] * (
                (1 - omega) * afacen[1] + omega * self.aface(i))
            if self.modeloptn == 0:
                rstor = 0.0
            elif self.modeloptn == 1:
                rstor = -fstor * (1 - omega) * c[i] + srate[i] * area[i] * \
                                                      delx[i] / twoplus * sc[i]
            else:
                rstor = -fstor * (1 - omega) * c[i] + epsilonr * area[i] * \
                                                      delx[i] / twoplus * sc[i]
            bvalue = dt * (sdt + sadv + sdisp + rstor)

        # interior segments
        else:
            # adv & disp
            if im1 is not None:
                sadv = -(1 - omega) * hadv * c[im1]  # bi-dir
                sdisp = -(1 - omega) * hdisp * c[im1]  # bi-dir
                bvalue = dt * (sadv + sdisp)
            elif ip1 is not None:
                sadv = -(1 - omega) * jadv * c[ip1]  # bi-dir
                sdisp = -(1 - omega) * jdisp * c[ip1]  # bi-dir
                bvalue = dt * (sadv + sdisp)  # bi-dir
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
                sdt = idt * c[i]  # bi-dir
                sadv = -(1 - omega) * iadv * c[i]  # bi-dir
                sdisp = -(1 - omega) * idisp * c[i]  # bi-dir
                bvalue = dt * (sdt + sadv + sdisp + rstor)  # bi-dir
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
