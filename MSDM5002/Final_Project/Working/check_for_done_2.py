def check_for_done(mat):
    done = False
    win = 0
    n = mat.shape[0]
    # always Draw for n<5
    if n<5:
        if (mat==0).sum() == 0:
            done = True
        return done, win
    # check horizontal and vertical
    i = 0
    while not done and i<n:
        for j in range(n-4):
            sequence_h = mat[i][j:j+5]
            sequence_v = mat[:,i][j:j+5]
            if sum(sequence_h)==5 or sum(sequence_h)==-5:
                done = True
                win = mat[i][j]
            if sum(sequence_v)==5 or sum(sequence_v)==-5:
                done = True
                win = mat[j][i]
        i+=1
    # check diagonal and anti-diagonal
    i = 0
    while not done and i<n-4:
        for j in range(n-4):
            sequence_diagonal = [mat[i+k][j+k] for k in range(5)]
            sequence_anti_diagonal = [mat[i+k][n-1-j-k] for k in range(5)]
            if sum(sequence_diagonal)==5 or sum(sequence_diagonal)==-5:
                done = True
                win = mat[i][j]
            if sum(sequence_anti_diagonal)==5 or sum(sequence_anti_diagonal)==-5:
                done = True
                win = mat[i][n-1-j]
        i+=1
    if not done and (mat==0).sum() == 0:
        done = True
    return done, win

# from numba import jit

# @jit(nopython=True)
def check_for_done_move(mat,move):
    n = mat.shape[0]
    if n<5:
        if (mat==0).sum() == 0:
            return True, 0
        return False, 0
    i = move[0]
    j = move[1]
    player = mat[i][j]
    offset = {'h':(0,1),'v':(1,0),'d':(1,1),'ad':(-1,1)}
    for os in offset.values():
        i1 = i + os[0]
        j1 = j + os[1]
        i2 = i - os[0]
        j2 = j - os[1]
        counting = 1
        while i1>=0 and j1>=0 and i1<n and j1<n:
            if mat[i1][j1]==player:
                counting+=1
                i1+=os[0]
                j1+=os[1]
            else:
                break
        while i2>=0 and j2>=0 and i2<n and j2<n:
            if mat[i2][j2]==player:
                counting+=1
                i2-=os[0]
                j2-=os[1]
            else:
                break
        if counting>=5:
            return True, player
    if (mat==0).sum() == 0:
        return True, 0
    return False, 0

    
    