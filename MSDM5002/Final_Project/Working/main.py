import pygame
import numpy as np
from check_for_done_2 import check_for_done_move
import MCTS_move

movecount = 0 # Keep track of movecount. It can be used when finding the children subset.
def update_by_man(event,mat):
    """
    This function detects the mouse click on the game window. Update the state matrix of the game.
    input:
        event:pygame event, which are either quit or mouse click)
        mat: 2D matrix represents the state of the game
    output:
        mat: updated matrix
    """
    global movecount
    done = False
    win = 0
    updated = False
    (row,col) = (0,0)
    d=int(560/(M-1))
    if event.type==pygame.QUIT:
        done=True
        win=99
    if event.type==pygame.MOUSEBUTTONDOWN:
        (x,y)=event.pos
        row = round((y - 40) / d)
        col = round((x - 40) / d)
        if mat[row][col] == 0:
            mat[row][col]=1
            updated = True
            movecount += 1
    return mat, done, win, updated, (row,col)

def update_by_pc(mat,move,movenumber):
    root = MCTS_move.GameNode(mat,None,1,move)
    chosen = MCTS_move.monte_carlo_tree_search(root, movenumber).move
    global movecount
    mat[chosen[0]][chosen[1]]=-1
    movecount += 1
    N_mat = np.zeros((M,M))
    ucb_mat = np.zeros((M,M))
    for child in root.children.values():
        i = child.move[0]
        j = child.move[1]
        N_mat[i][j] = child.N
        ucb_mat[i][j] = round(child.ucb(),3)
    print(f'Total simulation count = {root.N}')
    print(f'number of children considered = {len(root.children)}')
    print('children visited count:')
    print(N_mat)
    print('UCB matrix:')
    print(ucb_mat)
    return mat, chosen

def draw_board(screen):
    """
    This function draws the board with lines.
    input: game windows
    output: none
    """
    d=int(560/(M-1))
    black_color = [0, 0, 0]
    board_color = [241, 196, 15]
    screen.fill(board_color)
    for h in range(0, M):
        pygame.draw.line(screen, black_color,[40, h * d+40], [600, 40+h * d], 1)
        pygame.draw.line(screen, black_color, [40+d*h, 40], [40+d*h, 600], 1)
def draw_stone(screen, mat):
    """
    This functions draws the stones according to the mat. It draws a black circle for matrix element 1(human),
    it draws a white circle for matrix element -1 (computer)
    input:
        screen: game window, onto which the stones are drawn
        mat: 2D matrix representing the game state
    output:
        none
    """
    black_color = [0, 0, 0]
    white_color = [255, 255, 255]
    # M=len(mat)
    d=int(560/(M-1))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j]==1:
                pos = [40+d * j, 40+d* i ]
                pygame.draw.circle(screen, black_color, pos, 18,0)
            elif mat[i][j]==-1:
                pos = [40+d* j , 40+d * i]
                pygame.draw.circle(screen, white_color, pos, 18,0)

def render(screen, mat):
    """
    Draw the updated game with lines and stones using function draw_board and draw_stone
    input:
        screen: game window, onto which the stones are drawn
        mat: 2D matrix representing the game state
    output:
        none
    """
    draw_stone(screen, mat)
    pygame.display.update()
    

def main(board_size):
    
    global movecount
    global M
    M=board_size

    pygame.init()
    screen=pygame.display.set_mode((640,640))
    pygame.display.set_caption('Five-in-a-Row')
    done=False
    mat=np.zeros((M,M))
    draw_board(screen)
    pygame.display.update()

    while not done:
        for event in pygame.event.get():
            mat, done, win, updated, move = update_by_man(event,mat)
            render(screen, mat)
            if updated:
                done, win = check_for_done_move(mat,move)
                if done:
                    break
                mat, move = update_by_pc(mat,move, movecount)
                render(screen, mat)
                done, win = check_for_done_move(mat,move)
                if done:
                    break

    if win==0:
        print('Draw game')
    elif win==99:
        print('Cancelled game')
    elif win==1:
        print('You win the game!')
    else:
        print('You lose the game...')
    pygame.quit()
    
if __name__ == '__main__':
    main(8)
