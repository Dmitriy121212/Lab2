import numpy as np


def ABF0 (S0):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    YoutAB = np.zeros((iter, 1))
    T0=1
    for i in range(iter):
        Yin[i, 0] = float(S0[i])
    # -------------- початкові дані для запуску фільтра
    Yspeed_retro=(Yin[1, 0]-Yin[0, 0])/T0
    Yextra=Yin[0, 0]+Yspeed_retro
    alfa=2*(2*1-1)/(1*(1+1))
    beta=(6/1)*(1+1)
    YoutAB[0, 0]=Yin[0, 0]+alfa*(Yin[0, 0])
    # -------------- рекурентний прохід по вимірам
    for i in range(1, iter):
        if i >= 2000:

            YoutAB[i,0]=Yextra+alfa*(Yin[i, 0]- Yextra)
            Yspeed=Yspeed_retro+(beta/T0)*(Yin[i, 0]- Yextra)
            Yspeed_retro = Yspeed
            Yextra = YoutAB[i,0] + Yspeed_retro
            alfa = (2 * (2 * 2000 - 1)) / (2000 * (2000 + 1))
            beta = 6 /(2000* (2000 + 1))
        else:
            YoutAB[i, 0] = Yextra + alfa * (Yin[i, 0] - Yextra)
            Yspeed = Yspeed_retro + (beta / T0) * (Yin[i, 0] - Yextra)
            Yspeed_retro = Yspeed
            Yextra = YoutAB[i, 0] + Yspeed_retro
            alfa = (2 * (2 * i - 1)) / (i * (i + 1))
            beta = 6 / (i * (i + 1))
        # print('Регресійна модель ABF:')
        # print('i= ', i, '  y(t) = ', alfa, ' + ', beta, ' * t')
    print(YoutAB[1:20])
    return YoutAB

def ABGF(S0):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    YoutABG = np.zeros((iter, 1))
    T0 = 1
    for i in range(iter):
        Yin[i, 0] = float(S0[i])
    Yspeed_retro1 = (Yin[1, 0] - Yin[0, 0]) / T0
    Yspeed_retro2 = (Yin[2, 0] - Yin[1, 0]) / T0
    Yaccel_retro = Yspeed_retro2 - Yspeed_retro1 # Прискорення
    Vextra = Yspeed_retro1 + Yaccel_retro*T0
    Yextra = Yin[0, 0] + Yspeed_retro1*T0 + Yaccel_retro*T0/2

    alfa = 3 * (3 * 1 - 3 + 2) / ((1*(1 +1) * (1 + 2)))
    beta = 18*(2 - 1)/(T0*(2)*(3))
    gamma = 60/(1 * 2 * 3 * 1)
    YoutABG[0, 0] = Yin[0, 0] + alfa * (Yin[0, 0])
    for i in range(1, iter):
        YoutABG[i, 0] = Yextra + alfa * (Yin[i, 0] - Yextra)
        Yspeed = Vextra + (beta/T0)*(Yin[i, 0] - Yextra)
        Yaccel = Yaccel_retro + (gamma/(T0**2))*(Yin[i, 0] - Yextra)
        Yextra = YoutABG[i,0] + Yspeed*T0 + Yaccel*(T0/2)
        Vextra = Yspeed + Yaccel*T0
        Yaccel_retro = Yaccel
        alfa = (3*(3*(i**2) - 3*i + 2))/(i*(i + 1)*(i + 2))
        beta = (18*(2*i - 1))/(T0*i*(i+1)*(i + 2))
        gamma = 60/(i*(i+1)*(i+2))
        print("alfa = ", alfa)
        print("betta = ", beta)
        print("gamma = ", gamma)
    print(YoutABG)
    return YoutABG