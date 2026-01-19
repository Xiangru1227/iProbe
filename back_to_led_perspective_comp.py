

def perspective_comp(pcinfb, p1infp, fbinft, fpinft):
    '''
    pc: camera lens center in tracker frame, use nominal design
    p1: iProbe LED that needs to be corrected
    '''
    pcinft = fbinft @ pcinfb
    p1inft = fpinft @ p1infp
    
    '''compute p1_per (intersection of vector p1pc and plane y=0)'''
    p1_per = None # TODO: implement this
    p1_orth = 