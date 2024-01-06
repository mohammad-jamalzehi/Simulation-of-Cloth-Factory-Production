import numpy as np
import math
import matplotlib.pyplot as plt
from colorama import Fore, Style
from tabulate import tabulate
from scipy import stats
from scipy.stats import chi2_contingency

np.random.seed(42)

t_simulation_time = 24*60
warm_up = 4*24
simulation_time = t_simulation_time - warm_up
current_time = 0
FEL = [(0, "arrival_front"), (0, "arrival_back"), (0, 'arrival_collar')]


# Variables concerning queue
QRDM_history = []
QRDM_time_history = [0]
QLDM_history = []
QLDM_time_history = [0]
QP_history = []
QP_time_history = [0]
QS_history = []
QS_time_history = [0]
QPS_history = []
QPS_time_history = [0]
QJ_history = []
QJ_time_history = []
QSC_history = []
QSC_time_history = [0]
QI_history = []
QI_time_history = [0]
QI2_history = []
QI2_time_history = [0]
QCOR_history = []
QCOR_time_history = [0]
QF_history = []
QF_time_history = [0]
QPA_history = []
QPA_time_history = [0]
QRDM_avg = 0
QLDM_avg = 0
QP_avg = 0
QS_avg = 0
QPS_avg = 0
QJ_avg = 0
QSC_avg = 0
QI_avg = 0
QCOR_avg = 0
QI2_avg = 0
QF_avg = 0
QPA_avg = 0

# system Status
R = 0        # number of right parts of the shirt
L = 0        # number of left parts of the shirt
J = 0        # number of front parts of the shirt
B = 0        # number of back parts of the shirt
C = 0        # number of collar parts of the shirt
RDM = 0      # Right dose machine status
QRDM = 0     # Right dose machine Queue length
LDM = 0      # left dose machine status
QLDM = 0     # left dose machine Queue length
BP = 0       # button pierce machine status
QP = 0       # button pierce machine queue length
BS = 0       # button sewing machine status
QS = 0       # button sewing machine queue length
PS = 0       # pocket sewing machine status
QPS = 0      # pocket sewing machine queue length
JBF = 0      # back and front joint machine status
QJ = 0       # back and front joint machine queue length
SC = 0       # sleeve connector
QSC = 0      # sleeve connector queue length
INS = 0      # first inspection status
QI = 0       # inspection queue length
INS_2 = 0    # second inspection status
QI_2 = 0     # inspection queue length
IC = 0       # inspection counter
COR = 0      # correction machine status
QCOR = 0     # correction machine queue length
F = 0        # fold machine status
QF = 0       # fold machine queue length
PA = 0       # packing status
QPA = 0      # packing queue length
N = 0        # number of shirts
NSP = 0      # a dozen shirt
NW = 0       # number of wasted product
i = 0
# productivity report for each station
start_rdm = 0
prod_rdm = 0
start_ldm = 0
prod_ldm = 0
start_bp = 0
prod_bp = 0
start_bs = 0
prod_bs = 0
start_ps = 0
prod_ps = 0
start_jbf = 0
prod_jbf = 0
start_sc = 0
prod_sc = 0
start_ins = 0
prod_ins = 0
start_cor = 0
prod_cor = 0
start_ins2 = 0
prod_ins2 = 0
start_fold = 0
prod_fold = 0
start_pa = 0
prod_pa = 0
test = []

# Front Arrival event
def arrival_front():
    global current_time, RDM, LDM, QRDM, QLDM, start_rdm, start_ldm
    between_arrival_front = np.random.exponential(0.1192) * 60
    FEL.append((current_time + between_arrival_front, 'arrival_front'))
    FEL.sort()
    if RDM == 0:
        RDM = 1
        start_rdm = current_time
        RDM_duration = np.random.exponential(0.025) * 60
        FEL.append((current_time + RDM_duration, 'finish RDM'))
        FEL.sort()
    elif RDM == 1:
        QRDM += 1
        if current_time > warm_up:
            QRDM_history.append(QRDM)
            QRDM_time_history.append(current_time)
    if LDM == 0:
        LDM = 1
        if current_time > warm_up:
            start_ldm = current_time
        LDM_duration = np.random.exponential(0.025) * 60
        FEL.append((current_time + LDM_duration, 'finish LDM'))
        FEL.sort()
    elif LDM == 1:
        QLDM += 1
        if current_time > warm_up:
            QLDM_history.append(QLDM)
            QLDM_time_history.append(current_time)


# Right Dose Machine Finish Event
def finish_rdm():
    global current_time, QRDM, RDM, BS, QS, prod_rdm, start_bs
    if QRDM > 0:
        QRDM -= 1
        if current_time > warm_up:
            QRDM_history.append(QRDM)
            QRDM_time_history.append(current_time)
        RDM_duration = np.random.exponential(0.025) * 60
        FEL.append((current_time + RDM_duration, 'finish RDM'))
        FEL.sort()
    elif QRDM == 0:
        RDM -= 1
        if current_time >= warm_up:
            if warm_up > start_rdm:
                prod_rdm += current_time - warm_up
            else:
                prod_rdm += current_time - start_rdm
    if BS == 2:
        QS += 1
        if current_time > warm_up:
            QS_history.append(QS)
            QS_time_history.append(current_time)
    else:
        BS += 1
        start_bs = current_time
        BS_duration = np.random.exponential(0.040) * 60
        test.append(BS_duration)
        FEL.append((current_time + BS_duration, 'finish BS'))
        FEL.sort()


# Left Dose Machine Finish Event
def finish_ldm():
    global current_time, QLDM, LDM, BP, QP, start_bp, prod_ldm
    if QLDM > 0:
        QLDM -= 1
        if current_time > warm_up:
            QLDM_history.append(QLDM)
            QLDM_time_history.append(current_time)
        LDM_duration = np.random.exponential(0.025) * 60
        FEL.append((current_time + LDM_duration, 'finish LDM'))
        FEL.sort()
    elif QLDM == 0:
        LDM -= 1
        if current_time >= warm_up:
            if warm_up > start_ldm:
                prod_ldm += current_time - warm_up
            else:
                prod_ldm += current_time - start_ldm
    if BP == 2:
        QP += 1
        if current_time > warm_up:
            QP_history.append(QP)
            QP_time_history.append(current_time)
    else:
        BP += 1
        start_bp = current_time
        BP_duration = np.random.exponential(0.040) * 60
        FEL.append((current_time + BP_duration, 'finish BP'))
        FEL.sort()


# Button Sewing Finish Event
def finish_bs():
    global R, L, QS, BS, PS, QPS, prod_bs, start_ps, start_bs
    R += 1
    if QS > 0:
        QS -= 1
        if current_time > warm_up:
            QS_history.append(QS)
            QS_time_history.append(current_time)
        BS_duration = np.random.exponential(0.040) * 60
        test.append(BS_duration)
        FEL.append((current_time + BS_duration, 'finish BS'))
        FEL.sort()
    else:
        BS -= 1
        if current_time > warm_up:
            if start_bs < warm_up:
                prod_bs += (current_time - warm_up)
            else:
                prod_bs += (current_time - start_bs)
    if L > 0:
        L -= 1
        R -= 1
        if PS >= 4:
            QPS += 1
            if current_time > warm_up:
                QPS_history.append(QPS)
                QPS_time_history.append(current_time)
        elif PS < 4:
            PS += 1
            start_ps = current_time
            PS_duration = np.random.exponential(0.05) * 60
            FEL.append((current_time + PS_duration, 'finish PS'))
            FEL.sort()


# Button Pierce Finish Event
def finish_bp():
    global R, L, QP, BP, PS, QPS, prod_bp, start_ps, start_bp
    L += 1
    if QP > 0:
        QP -= 1
        if current_time > warm_up:
            QP_history.append(QP)
            QP_time_history.append(current_time)
        BP_duration = np.random.exponential(0.040) * 60
        FEL.append((current_time + BP_duration, 'finish BP'))
        FEL.sort()
    else:
        BP -= 1
        if current_time >= warm_up:
            if warm_up > start_bp:
                prod_bp += current_time - warm_up
            else:
                prod_bp += current_time - start_bp
    if R > 0:
        L -= 1
        R -= 1
        if PS >= 4:
            QPS += 1
            if current_time > warm_up:
                QPS_history.append(QPS)
                QPS_time_history.append(current_time)
        elif PS < 4:
            PS += 1
            start_ps = current_time
            PS_duration = np.random.exponential(0.05) * 60
            FEL.append((current_time + PS_duration, 'finish PS'))
            FEL.sort()


# Pocket Sewing Finish Event
def finish_ps():
    global J, QPS, PS, B, C, JBF, QJ, i, prod_ps, start_jbf
    J += 1
    if QPS > 0:
        QPS -= 1
        if current_time > warm_up:
            QPS_history.append(QPS)
            QPS_time_history.append(current_time)
        PS_duration = np.random.exponential(0.05) * 60
        FEL.append((current_time + PS_duration, 'finish PS'))
        FEL.sort()
    elif QPS == 0:
        PS -= 1
        if current_time > warm_up:
            prod_ps += current_time - start_ps
    if B > 0 and C > 0:
        B -= 1
        C -= 1
        J -= 1
        if JBF == 0:
            JBF += 1
            start_jbf = current_time
            JBF_duration = np.random.exponential(0.071) * 60
            JBFC_duration = np.random.exponential(0.1) * 60
            JBF_duration_max = max(JBF_duration, JBFC_duration)
            FEL.append((current_time + JBF_duration_max, 'finish JBF'))
            FEL.sort()
        elif JBF == 1:
            QJ += 1
            if current_time > warm_up:
                QJ_history.append(QJ)
                QJ_time_history.append(current_time)


# Back Arrival Event
def arrival_back():
    global B, J, C, JBF, QJ, start_jbf
    between_arrival_back = np.random.exponential(0.1192) * 60
    FEL.append((current_time+between_arrival_back, "arrival_back"))
    FEL.sort()
    B += 1
    if J > 0 and C > 0:
        J -= 1
        B -= 1
        C -= 1
        if JBF == 0:
            JBF += 1
            start_jbf = current_time
            JBF_duration = np.random.exponential(0.071) * 60
            JBFC_duration = np.random.exponential(0.1) * 60
            JBF_duration_max = max(JBF_duration, JBFC_duration)
            FEL.append((current_time+JBF_duration_max, 'finish JBF'))
            FEL.sort()
        elif JBF == 1:
            QJ += 1
            if current_time > warm_up:
                QJ_history.append(QJ)
                QJ_time_history.append(current_time)


# Collar Arrival Event
def arrival_collar():
    global B, J, C, JBF, QJ, start_jbf
    between_arrival_collar = np.random.exponential(0.1192) * 60
    FEL.append((current_time + between_arrival_collar, "arrival_collar"))
    FEL.sort()
    C += 1
    if J > 0 and B > 0:
        J -= 1
        B -= 1
        C -= 1
        if JBF == 0:
            JBF += 1
            start_jbf = current_time
            JBF_duration = np.random.exponential(0.071) * 60
            JBFC_duration = np.random.exponential(0.1) * 60
            JBF_duration_max = max(JBF_duration, JBFC_duration)
            FEL.append((current_time + JBF_duration_max, 'finish JBF'))
            FEL.sort()
        elif JBF == 1:
            QJ += 1
            if current_time > warm_up:
                QJ_history.append(QJ)
                QJ_time_history.append(current_time)


# Back and Front Joint Finish Event
def finish_jbf():
    global QJ, JBF, SC, QSC, prod_jbf, start_sc
    if QJ > 0:
        QJ -= 1
        if current_time > warm_up:
            QJ_history.append(QJ)
            QJ_time_history.append(current_time)
        JBF_duration = np.random.exponential(0.071) * 60
        JBFC_duration = np.random.exponential(0.1) * 60
        JBF_duration_max = max(JBF_duration, JBFC_duration)
        FEL.append((current_time + JBF_duration_max, 'finish JBF'))
        FEL.sort()
    else:
        JBF = 0
        if current_time > warm_up:
            prod_jbf += current_time - start_jbf
    if SC == 2:
        QSC += 1
        if current_time > warm_up:
            QSC_history.append(QSC)
            QSC_time_history.append(current_time)
    else:
        SC += 1
        start_sc = current_time
        SC_duration = np.random.exponential(0.5) * 60
        FEL.append((current_time + SC_duration, 'finish SC'))
        FEL.sort()


# Sleeve Connector Finish Event
def finish_sc():
    global QSC, SC, INS, QI, IC, prod_sc, start_ins
    if QSC > 0:
        QSC -= 1
        if current_time > warm_up:
            QSC_history.append(QSC)
            QSC_time_history.append(current_time)
        SC_duration = np.random.exponential(0.5) * 60
        FEL.append((current_time + SC_duration, 'finish SC'))
        FEL.sort()
    else:
        SC -= 1

    if INS == 7:
        QI += 1
        if current_time > warm_up:
            QI_history.append(QI)
            QI_time_history.append(current_time)
    else:
        INS += 1
        start_ins = current_time
        INS_duration = np.random.exponential(0.125) * 60
        FEL.append((current_time + INS_duration, 'finish INS'))
        FEL.sort()


# Inspection Finish Event
def finish_ins():
    global F, QF, COR, QCOR, N, QI, INS, start_cor, start_fold, prod_ins
    if QI > 0:
        QI -= 1
        if current_time >= warm_up:
            QI_history.append(QI)
            QI_time_history.append(current_time)
        INS_duration = np.random.exponential(0.125) * 60
        FEL.append((current_time + INS_duration, 'finish INS'))
        FEL.sort()
    elif QI == 0:
        INS -= 1
        if current_time > warm_up:
            prod_ins += current_time - start_ins
    r = np.random.uniform(0, 1)
    if 0 < r <= 0.03:
        if COR == 0:
            COR += 1
            start_cor = current_time
            Correction_duration = np.random.exponential(0.1) * 60
            FEL.append((current_time + Correction_duration, 'finish COR'))
            FEL.sort()
        elif COR == 1:
            QCOR += 1
            if current_time > warm_up:
                QCOR_history.append(QCOR)
                QCOR_time_history.append(current_time)
    else:
        if F == 0:
            F = 1
            start_fold = current_time
            fold_duration = np.random.exponential(0.21) * 60
            FEL.append((current_time + fold_duration, 'finish fold'))
            FEL.sort()
        else:
            QF += 1
            if current_time > warm_up:
                QF_history.append(QF)
                QF_time_history.append(current_time)


# Correction Finish Event
def finish_cor():
    global QCOR, COR, INS_2, QI_2, start_ins2, prod_cor
    if QCOR > 0:
        QCOR -= 1
        if current_time > warm_up:
            QCOR_history.append(QCOR)
            QCOR_time_history.append(current_time)
        Correction_duration = np.random.exponential(0.1) * 60
        FEL.append((current_time + Correction_duration, 'finish COR'))
        FEL.sort()
    elif QCOR == 0:
        COR = 0
        if current_time > warm_up:
            prod_cor += current_time - start_cor
    if INS_2 == 1:
        QI_2 += 1
        if current_time > warm_up:
            QI2_history.append(QI_2)
            QI2_time_history.append(current_time)
    elif INS_2 == 0:
        INS_2 += 1
        start_ins2 = current_time
        INS2_duration = np.random.exponential(0.125) * 60
        FEL.append((current_time + INS2_duration, 'finish INS_2'))
        FEL.sort()


# Second Inspection Finish Event
def finish_ins2():
    global NW, QF, F, QI_2, INS_2, N, prod_ins2, start_fold
    if QI_2 > 0:
        QI_2 -= 1
        if current_time > warm_up:
            QI2_history.append(QI_2)
            QI2_time_history.append(current_time)
        INS2_duration = np.random.exponential(0.125) * 60
        FEL.append((current_time + INS2_duration, 'finish INS_2'))
        FEL.sort()
    else:
        INS_2 = 0
        if current_time > warm_up:
            prod_ins2 += current_time - start_ins2
    r = np.random.uniform(0, 1)
    if 0 < r <= 0.03:
        if current_time > warm_up:
            NW += 1
    else:
        if F == 1:
            QF += 1
            if current_time > warm_up:
                QF_history.append(QF)
                QF_time_history.append(current_time)
        elif F == 0:
            F = 1
            start_fold = current_time
            fold_duration = np.random.exponential(0.21) * 60
            FEL.append((current_time + fold_duration, 'finish fold'))
            FEL.sort()


# Fold Finish Event
def finish_fold():
    global QF, F, N, NSP, PA, QPA, prod_fold, start_pa
    N += 1
    if QF > 0:
        QF -= 1
        if current_time > warm_up:
            QF_history.append(QF)
            QF_time_history.append(current_time)
        fold_duration = np.random.exponential(0.21) * 60
        FEL.append((current_time + fold_duration, 'finish fold'))
        FEL.sort()
    elif QF == 0:
        F = 0
        if current_time > warm_up:
            prod_fold += current_time - start_fold
    if N >= 12:
        N = N - 12
        if PA == 0:
            PA = 1
            start_pa = current_time
            packing_duration = np.random.exponential(0.02) * 60
            FEL.append((current_time + packing_duration, 'finish packing'))
            FEL.sort()
        else:
            QPA += 1
            if current_time > warm_up:
                QPA_history.append(QPA)
                QPA_time_history.append(current_time)


# Packing Finish Event
def finish_packing():
    global QPA, PA, NSP, N, prod_pa
    if current_time > warm_up:
        NSP += 12
    if QPA > 0:
        QPA -= 1
        if current_time > warm_up:
            QPA_history.append(QPA)
            QPA_time_history.append(current_time)
        packing_duration = np.random.exponential(0.02) * 60
        FEL.append((current_time + packing_duration, 'finish packing'))
        FEL.sort()
    else:
        PA = 0
        if current_time > warm_up:
            prod_pa += current_time - start_pa


while current_time < t_simulation_time:
    current_time, event_type = FEL.pop(0)
    if event_type == 'arrival_front':
        arrival_front()
    elif event_type == 'arrival_back':
        arrival_back()
    elif event_type == 'arrival_collar':
        arrival_collar()
    elif event_type == 'finish RDM':
        finish_rdm()
    elif event_type == 'finish LDM':
        finish_ldm()
    elif event_type == 'finish BP':
        finish_bp()
    elif event_type == 'finish BS':
        finish_bs()
    elif event_type == 'finish PS':
        finish_ps()
    elif event_type == 'finish JBF':
        finish_jbf()
    elif event_type == 'finish SC':
        finish_sc()
    elif event_type == 'finish INS':
        finish_ins()
    elif event_type == 'finish COR':
        finish_cor()
    elif event_type == 'finish INS_2':
        finish_ins2()
    elif event_type == 'finish fold':
        finish_fold()
    elif event_type == 'finish packing':
        finish_packing()

# Calculating queue average
QRDM_time_history.append(simulation_time)
QRDM_history.append(QRDM)
for i in range(len(QRDM_time_history)-1):
    QRDM_avg += (QRDM_time_history[i + 1] - QRDM_time_history[i]) * QRDM_history[i]
QRDM_avg = QRDM_avg / simulation_time


QLDM_time_history.append(simulation_time)
QLDM_history.append(QLDM)
for i in range(len(QLDM_time_history)-1):
    QLDM_avg += (QLDM_time_history[i + 1] - QLDM_time_history[i]) * QLDM_history[i]
QLDM_avg = QLDM_avg / simulation_time


QS_time_history.append(simulation_time)
QS_history.append(QS)
for i in range(len(QS_time_history)-1):
    QS_avg += (QS_time_history[i + 1] - QS_time_history[i]) * QS_history[i]
QS_avg = QS_avg / simulation_time


QP_time_history.append(simulation_time)
QP_history.append(QP)
for i in range(len(QP_time_history)-1):
    QP_avg += (QP_time_history[i + 1] - QP_time_history[i]) * QP_history[i]
QP_avg = QP_avg / simulation_time


QPS_time_history.append(simulation_time)
QPS_history.append(QPS)
for i in range(len(QPS_time_history)-1):
    QPS_avg += (QPS_time_history[i + 1] - QPS_time_history[i]) * QPS_history[i]
QPS_avg = QPS_avg / simulation_time


QJ_time_history.append(simulation_time)
QJ_history.append(QJ)
for i in range(len(QJ_time_history)-1):
    QJ_avg += (QJ_time_history[i + 1] - QJ_time_history[i]) * QJ_history[i]
QJ_avg = QJ_avg / simulation_time


QSC_time_history.append(simulation_time)
QSC_history.append(QSC)
for i in range(len(QSC_time_history)-1):
    QSC_avg += (QSC_time_history[i + 1] - QSC_time_history[i]) * QSC_history[i]
QSC_avg = QSC_avg / simulation_time


QI_time_history.append(simulation_time)
QI_history.append(QI)
for i in range(len(QI_time_history)-1):
    QI_avg += (QI_time_history[i + 1] - QI_time_history[i]) * QI_history[i]
QI_avg = QI_avg / simulation_time


QCOR_time_history.append(simulation_time)
QCOR_history.append(QCOR)
for i in range(len(QCOR_time_history)-1):
    QCOR_avg += (QCOR_time_history[i + 1] - QCOR_time_history[i]) * QCOR_history[i]
QCOR_avg = QCOR_avg / simulation_time


QI2_time_history.append(simulation_time)
QI2_history.append(QI_2)
for i in range(len(QI2_time_history)-1):
    QI2_avg += (QI2_time_history[i + 1] - QI2_time_history[i]) * QI2_history[i]
QI2_avg = QI2_avg / simulation_time


QF_time_history.append(simulation_time)
QF_history.append(QF)
for i in range(len(QF_time_history)-1):
    QF_avg += (QF_time_history[i + 1] - QF_time_history[i]) * QF_history[i]
QF_avg = QF_avg / simulation_time


QPA_time_history.append(simulation_time)
QPA_history.append(QPA)
for i in range(len(QPA_time_history)-1):
    QPA_avg += (QPA_time_history[i + 1] - QPA_time_history[i]) * QPA_history[i]
QPA_avg = QPA_avg / simulation_time


# Calculating productivity
if RDM > 0:
    prod_rdm += simulation_time - start_rdm
if LDM > 0:
    prod_ldm += simulation_time - start_ldm
if BS > 0:
    prod_bs += simulation_time - start_bs
if BP > 0:
    prod_bp += simulation_time - start_bp
if PS > 0:
    prod_ps += simulation_time - start_ps
if JBF > 0:
    prod_jbf += simulation_time - start_jbf
if SC > 0:
    prod_sc += simulation_time - start_sc
if INS > 1:
    prod_ins += simulation_time - start_ins
if COR > 1:
    prod_cor += simulation_time - start_cor
if INS_2 == 1:
    prod_ins2 += simulation_time - start_ins2
if F == 1:
    prod_fold += simulation_time - start_fold
if PA == 1:
    prod_pa += simulation_time - start_pa

print(f"{Style.BRIGHT}{Fore.LIGHTBLUE_EX} First output= 1-queue length history for each section : {Style.RESET_ALL}")
print('Right dose machine queue history : ', QRDM_history)
print('Left dose machine queue history : ', QLDM_history)
print('Button Sewing machine queue history : ', QS_history)
print('Button pierce machine queue history : ', QP_history)
print('Pocket sewing machine queue history : ', QPS_history)
print('Back and front joint machine queue history : ', QJ_history)
print('Sleeve connector queue history : ', QSC_history)
print('Inspection queue history : ', QI_history)
print('Correction machine queue history : ', QCOR_history)
print('Second inspection queue history : ', QI2_history)
print('Fold machine queue history : ', QF_history)
print('Packing queue history : ', QPA_history)
print('\n')
print(f"{Style.BRIGHT}{Fore.LIGHTBLUE_EX} First output= 2-queue average length for each section : {Style.RESET_ALL}")
print('Right dose machine average queue : ', np.round(QRDM_avg, 4))
print('Left dose machine average queue : ', np.round(QLDM_avg, 4))
print('Button Sewing machine average queue : ', np.round(QS_avg, 4))
print('Button pierce machine average queue : ', np.round(QP_avg, 4))
print('Pocket sewing machine average queue : ', np.round(QPS_avg, 4))
print('Back and front joint machine average queue : ', np.round(QJ_avg, 4))
print('Sleeve connector average queue : ', np.round(QSC_avg, 4))
print('Inspection average queue : ', np.round(QI_avg, 4))
print('Correction machine average queue : ', np.round(QCOR_avg, 4))
print('Second inspection average queue : ', np.round(QI2_avg, 4))
print('Fold machine average queue : ', np.round(QF_avg, 4))
print('Packing average queue : ', np.round(QPA_avg, 4))
print('\n')
print(f"{Style.BRIGHT}{Fore.LIGHTMAGENTA_EX} Second output = Number of Wasting  : {Style.RESET_ALL}")
print("The number of wasted products : ", NW)
print('\n')
# print(f"{Style.BRIGHT}{Fore.LIGHTGREEN_EX} Third output = Efficiency  : {Style.RESET_ALL}")
# print(f'product quantity in {simulation_time/60} hours of production: {NSP}')
# print('productivity of RDM station:', float(int((prod_rdm / simulation_time)*100)), '%')
# print('productivity of LDM station:', round((prod_ldm / simulation_time)*100, 2), '%')
# print('productivity of BS station:', float(int((prod_rdm / simulation_time)*100)), '%')
# print('productivity of BP station:', float(int((prod_rdm / simulation_time)*100)), '%')
# print('productivity of PS station:', round((prod_ps / simulation_time)*100, 2), '%')
# print('productivity of JBF station:', round((prod_jbf / simulation_time)*100, 2), '%')
# print('productivity of SC station:', round((prod_sc / simulation_time)*100, 2), '%')
# print('productivity of INS station:', round((prod_ins / simulation_time)*100, 2), '%')
# print('productivity of COR station:', round((prod_cor / simulation_time)*100, 2), '%')
# print('productivity of INS2 station:', round((prod_ins2 / simulation_time)*100, 2), '%')
# print('productivity of FOLD station:', round((prod_fold / simulation_time)*100, 2), '%')
# print('productivity of PACKING station:', round((prod_pa / simulation_time)*100, 2), '%')
print(f"{Style.BRIGHT}{Fore.LIGHTGREEN_EX} Third output = Efficiency  : {Style.RESET_ALL}", "")
data = [
    ("RDM station", f'{float(int((prod_rdm / simulation_time)*100))}%'),
    ("LDM station", f'{round((prod_ldm / simulation_time)*100, 2)}%'),
    ("BS station", f'{float(int((prod_rdm / simulation_time)*100))}%'),
    ("BP station", f'{float(int((prod_rdm / simulation_time)*100))}%'),
    ("PS station", f'{round((prod_ps / simulation_time)*100, 2)}%'),
    ("JBF station", f'{round((prod_jbf / simulation_time)*100, 2)}%'),
    ("SC station", f'{round((prod_sc / simulation_time)*100, 2)}%'),
    ("INS station", f'{round((prod_ins / simulation_time)*100, 2)}%'),
    ("COR station", f'{round((prod_cor / simulation_time)*100, 2)}%'),
    ("INS2 station", f'{round((prod_ins2 / simulation_time)*100, 2)}%'),
    ("FOLD station", f'{round((prod_fold / simulation_time)*100, 2)}%'),
    ("PACKING station", f'{round((prod_pa / simulation_time)*100, 2)}%'),
]

print(tabulate(data, headers=["Station", "productivity"], tablefmt="fancy_grid"))
