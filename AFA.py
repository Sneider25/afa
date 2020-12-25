# Adaptive Fractal Analysis, (c) 2020 Jordi
import numpy as np
import matplotlib.pyplot as plt

def AFA(x, w, d=1, draw=False):
    w = int(w)
    # Enforce even windows
    if w % 2 != 0:
        # print('Warning, incrementing w')
        w += 1
    n = int((w-1)/2)
    N = len(x)
    # Windows start and end
    wstart = np.arange(0, N, n+1)
    wend = wstart + w - 1
    # Make sure wend does not point further than N=len(x)
    wend = np.min((np.repeat(N-1, len(wend)), wend), axis=0)

    # Weights
    l = np.arange(n+1, dtype='int') + 1
    w1 = 1 - (l-1)/n
    # w2 is 1-w1

    if draw:
        ejex = np.arange(N) + 1
        plt.plot(ejex, x)

    yf = np.zeros(x.shape)
    for i in range(len(wstart)-1):
        # print(wstart[i], wend[i])
        t1 = np.arange(wstart[i], wend[i]+1, dtype='int')
        t2 = np.arange(wstart[i+1], wend[i+1]+1, dtype='int')
        # Polynomial adjustment
        p1 = np.polyfit(t1, x[t1], d)
        p2 = np.polyfit(t2, x[t2], d)
        y1 = np.polyval(p1, t1)
        y2 = np.polyval(p2, t2)

        # In the last steps this may happen. Enlarge y1 and y2 to w
        if len(y1) < w:
            y1 = np.concatenate((y1, np.zeros(w-len(y1))))
        if len(y2) < w:
            y2 = np.concatenate((y2, np.zeros(w-len(y2))))

        # Adjust overlapping points
        yc = w1 * y1[(l+n)] + (1-w1) * y2[(l-1)]

        # Overlapping area (n+1 length)
        tt = np.arange(t2[0], np.min((N, t2[0] + n+1)), dtype='int')
        # Ensure len(yc) = len(tt)
        yc = yc[:len(tt)]
        yf[tt] = yc.reshape(-1,1)

        # First non-overlapping area
        if i == 0:
            yf[:n+1] = y1[:n+1].reshape(-1,1)

        if draw:
            plt.plot(t1, y1[:len(t1)], 'k')
            plt.plot(t2, y2[:len(t2)], 'k')
            plt.plot(tt, yc, 'r')

    # Residuals RMSE
    f = np.sqrt(np.sum((x - yf)**2) / N)
    return f, yf