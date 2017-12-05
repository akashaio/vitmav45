from libc cimport math
import cython
import numpy as np
cimport numpy as np
np.seterr(divide='raise',invalid='raise')

ctypedef np.float64_t f_t

# Turn off bounds checking, negative indexing
@cython.boundscheck(False)
@cython.wraparound(False)
def ctc_loss(double[::1,:] params not None, int[::1] seq not None, unsigned int blank=0):
    """
    CTC loss function.
    Forrás: https://github.com/amaas/stanford-ctc/blob/master/ctc_fast/ctc-loss/ctc_fast.pyx
    params - n x m mátrix, m keretre n dimenziós valószínűség eloszlás a beszédhangokra. Memóriába várjuk, Fortran order-ben.
    seq - az adott hangmintának megfelelő beszédhang-szekvencia
    Visszatérési értékek: a loss értéke és a gradiensek
    """

    cdef unsigned int seqLen = seq.shape[0] # a beszédhang-szekvencia hossza (# phones)
    cdef unsigned int numphones = params.shape[0] # beszédhnagok száma (ennyit különböztetünk meg)
    cdef unsigned int L = 2*seqLen + 1 # blankokkal bővített szekvenciahossz
    cdef unsigned int T = params.shape[1] # Időkeretek száma (time)

    cdef double[::1,:] alphas = np.zeros((L,T), dtype=np.double, order='F') # Alfa mátrix
    cdef double[::1,:] betas = np.zeros((L,T), dtype=np.double, order='F') # Béta mátrix
    cdef double[::1,:] ab = np.empty((L,T), dtype=np.double, order='F') # alfa * béta
    cdef np.ndarray[f_t, ndim=2] grad = np.zeros((numphones,T),  # gradiens
                            dtype=np.double, order='F')
    cdef double[::1,:] grad_v = grad
    cdef double[::1] absum = np.empty(T, dtype=np.double) # alfa*béta kumulatív szorzata (külső szummához)

    # Segédváltozók
    cdef unsigned int start, end
    cdef unsigned int t, s, l
    cdef double c, llForward, llBackward, llDiff, tmp

    try:
        # Forward irány, alfák inicializálása és rekurzív számítása
        alphas[0,0] = params[blank,0]
        alphas[1,0] = params[seq[0],0]
        c = alphas[0,0] + alphas[1,0]
        alphas[0,0] = alphas[0,0] / c
        alphas[1,0] = alphas[1,0] / c
        llForward = math.log(c)
        for t in xrange(1,T):
            start = 2*(T-t) # Ennyi lépésünk lehet még (a kétszeres szorzó azért kell, mert a blankok opcionálisak, azaz át is ugorhatjuk őket)
            if L <= start: # Ha van még annyi lépésünk, hogy 0-tól L-ig bejárjuk címkeszekvenciát a maradék idő alatt, akkor a start 0, különben felesleges vele számolni
                start = 0
            else:
                start = L-start # L címke van összesen (blankokkal együtt)
            end = min(2*t+2,L)
            for s in xrange(start,L):
                l = (s-1)/2
                # Ha blank-ba megyünk (0., majd minden páros index)
                if s%2 == 0:
                    if s==0:
                        alphas[s,t] = alphas[s,t-1] * params[blank,t] # Helyben maradás
                    else:
                        alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[blank,t] # Helyben maradás, vagy az előző állapotból való átlépés
                # Ha helyben maradunk (vagy annyira elől tartunk, hogy nem volt előző nem blank állapotunk)
                elif s == 1 or seq[l] == seq[l-1]:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t]
                # Minden más eset (előre lépünk egyet vagy kettőt, utóbbi, ha blankot ugrunk át)
                else:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) * params[seq[l],t]
                
            # normalizálás (ne csorduljon alul)
            c = 0.0
            for s in xrange(start,end):
                c += alphas[s,t]
            for s in xrange(start,end):
                alphas[s,t] = alphas[s,t] / c
            llForward += math.log(c)

        # Backward irány
        betas[L-1,T-1] = params[blank,T-1]
        betas[L-2,T-1] = params[seq[seqLen-1],T-1]
        c = betas[L-1,T-1] + betas[L-2,T-1]
        betas[L-1,T-1] = betas[L-1,T-1] / c
        betas[L-2,T-1] = betas[L-2,T-1] / c
        llBackward = math.log(c)
        for t in xrange(T-1,0,-1):
            t = t-1
            start = 2*(T-t)
            if L <= start:
                start = 0
            else:
                start = L-start
            end = min(2*t+2,L)
            for s in xrange(end,0,-1):
                s = s-1
                l = (s-1)/2
                # Ha blank-ba megyünk (0., majd minden páros index)
                if s%2 == 0:
                    if s == L-1:
                        betas[s,t] = betas[s,t+1] * params[blank,t]
                    else:
                        betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[blank,t]
                # Ha helyben maradunk (vagy nem volt előző nem blank állapotunk)
                elif s == L-2 or seq[l] == seq[l+1]:
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t]
                # Minden más eset (előre lépünk egyet vagy kettőt, utóbbi, ha blankot ugrunk át)
                else:
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) * params[seq[l],t]

            c = 0.0
            for s in xrange(start,end):
                c += betas[s,t]
            for s in xrange(start,end):
                betas[s,t] = betas[s,t] / c
            llBackward += math.log(c)

        # A gradiens számítása (a nem normalizált paraméterek alapján, azaz softmax előtti esetre)
        for t in xrange(T):
            for s in xrange(L):
                ab[s,t] = alphas[s,t]*betas[s,t] # A szorzat minden rácspontra (alfa_t*beta_t)
        for s in xrange(L):
            # blank eset és nem blank eset
            if s%2 == 0:
                for t in xrange(T):
                    grad_v[blank,t] += ab[s,t]
                    if ab[s,t] != 0:
                        ab[s,t] = ab[s,t]/params[blank,t] # 1/ y^k_t
            else:
                for t in xrange(T):
                    grad_v[seq[(s-1)/2],t] += ab[s,t]
                    if ab[s,t] != 0:
                        ab[s,t] = ab[s,t]/(params[seq[(s-1)/2],t]) # 1/ y^k_t

        for t in xrange(T):
            absum[t] = 0
            for s in xrange(L):
                absum[t] += ab[s,t] # Szummázzuk 1-re normáláshoz

        # grad = params - grad / (params * absum)
        for t in xrange(T):
            for s in xrange(numphones):
                tmp = (params[s,t]*absum[t])
                if tmp > 0:
                    grad_v[s,t] = params[s,t] - grad_v[s,t] / tmp 
                else:
                    grad_v[s,t] = params[s,t]

    except (FloatingPointError,ZeroDivisionError) as e:
        print e.message
        return -llForward,grad,True


    return -llForward,grad,False

def decode_best_path(double[::1,:] probs not None, unsigned int blank=0):
    """
    Computes best path given sequence of probability distributions per frame.
    Simply chooses most likely label at each timestep then collapses result to
    remove blanks and repeats.
    Optionally computes edit distance between reference transcription
    and best path if reference provided.
    """

    # Compute best path
    cdef unsigned int T = probs.shape[1]
    cdef long [::1] best_path = np.argmax(probs,axis=0)

    # Collapse phone string
    cdef unsigned int i, b
    hyp = []
    align = []
    for i in xrange(T):
        b = best_path[i]
        # ignore blanks
        if b == blank:
            continue
        # FIXME ignore some special characters
        # noise, laughter, vocalized-noise
        if b == 1 or b == 2 or b == 8:
            continue
        # ignore repeats
        elif i != 0 and  b == best_path[i-1]:
            align[-1] = i
            continue
        else:
            hyp.append(b)
            align.append(i)
    return hyp, align

