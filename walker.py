import random
import numpy as np

from collections import deque

trap_test_threshhold = 100

class Moves:
    UP = 'Up'
    RIGHT = 'Right'
    DOWN = 'Down'
    LEFT = 'Left'

    STOP = 'Stop'

    moves =    [UP,
                RIGHT,
                DOWN,
                LEFT]

    deltas =  { UP      : np.array([0,1]),
                RIGHT   : np.array([1,0]),
                DOWN    : np.array([0,-1]),
                LEFT    : np.array([-1,0]),
                STOP    : np.array([0,0])  }

    reverse = { UP      : DOWN,
                RIGHT   : LEFT,
                DOWN    : UP,
                LEFT    : RIGHT,
                STOP    : STOP }

class BasicSAW(object):
    def __init__(self, start=(0,0), up=0.25, right=0.25, down=0.25, left=0.25):
        self.pos = np.array(start)

        self.probs = {  Moves.UP    : up,
                        Moves.RIGHT : right,
                        Moves.DOWN  : down,
                        Moves.LEFT  : left  }

        self.nSteps = 0
        self.visited = [start]

    def walk(self):
        rando = random.random()
        prob_threshold = 0
        for move in Moves.moves:
            prob_threshold += self.probs[move]
            if rando < prob_threshold:
                return self.apply_move(move)

    def apply_move(self,move,test=False):
        new_xy = self.pos + Moves.deltas[move]

        '''
        print x,y
        print new_xy
        print self.visited

        raw_input('press enter...')
        '''

        if tuple(new_xy) in self.visited:
            return -1

        if not test:
            self.pos = new_xy
            self.visited.append(tuple(new_xy))
            self.nSteps += 1

        return 0

    def renormalize(self,possible_moves):
        possible_probs = np.array([self.probs[move] for move in possible_moves])
        possible_probs /= float(sum(possible_probs))

        probs = {}
        for i in range(len(possible_moves)):
            probs.update({possible_moves[i] : possible_probs[i]})

        return probs

    def getR2(self):
        x,y = tuple(self.pos)
        sx,sy = self.visited[0]
        return (x-sx)**2 + (y-sy)**2

class SmartSAW(BasicSAW):
    def __init__(self, start=(0,0), up=0.25, right=0.25, down=0.25, left=0.25):
        super(SmartSAW,self).__init__(start,up,right,down,left)
        self.prev_move = Moves.STOP

    def walk(self):
        rando = random.random()
        backwards = Moves.reverse[self.prev_move]
        possible_moves = Moves.moves[:]

        # knows that it can't go backwards
        if not backwards == Moves.STOP:
            possible_moves.remove(backwards)

        renormalized_probs = self.renormalize(possible_moves)

        prob_threshold = 0
        for move in possible_moves:
            prob_threshold += renormalized_probs[move]
            if rando < prob_threshold:
                self.prev_move = move
                return self.apply_move(move)

        # some error occured
        return -1

class SmarterSAW(BasicSAW):
    def __init__(self, start=(0,0), up=0.25, right=0.25, down=0.25, left=0.25):
        super(SmarterSAW,self).__init__(start,up,right,down,left)

    def walk(self):
        rando = random.random()
        possible_moves = [move for move in Moves.moves if self.apply_move(move,test=True) == 0]

        # trapped
        if len(possible_moves) == 0:
            return -1

        renormalized_probs = self.renormalize(possible_moves)

        prob_threshold = 0
        for move in possible_moves:
            prob_threshold += renormalized_probs[move]
            if rando < prob_threshold:
                return self.apply_move(move)

        # some other error occured
        return -1

class SmartestSAW(BasicSAW):
    def __init__(self, start=(0,0), up=0.25, right=0.25, down=0.25, left=0.25):
        super(SmartestSAW,self).__init__(start,up,right,down,left)

    def walk(self):
        rando = random.random()
        possible_moves = [move for move in Moves.moves if self.apply_move(move,test=True) == 0]

        # trap avoidance
        # Note -- could improve efficiency by only checking for traps in certain situations
        possible_moves = [move for move in possible_moves if not self.isTrap(move)]

        renormalized_probs = self.renormalize(possible_moves)

        prob_threshold = 0
        for move in possible_moves:
            prob_threshold += renormalized_probs[move]
            if rando < prob_threshold:
                return self.apply_move(move)

        # some other error occured
        return -1

    def isTrap(self,move):
        start_pos = self.pos + Moves.deltas[move]
        visited = self.visited[:]

        #print move

        # BFS
        q = deque()
        q.append(tuple(start_pos))
        nClear = 0
        while len(q) > 0:
            #print q
            pos = np.array(q.popleft())
            visited.append(tuple(pos))
            nClear += 1

            for next_move in Moves.moves:
                new_pos = pos + Moves.deltas[next_move]
                if not (tuple(new_pos) in visited or tuple(new_pos) in q):
                    q.append(tuple(new_pos))
            #print q

            #raw_input('ssdfs')

            # can be assumed that infinitely many more spaces can be reached --> not trapped
            if nClear > trap_test_threshhold:
                return False

        # queue exhausted --> trapped
        return True

def allDistances(n):
    return recursiveHelper(n,[(0,0)])

def recursiveHelper(level,visited):
    x,y = visited[-1]
    if level == 0:
        # euclidean distance of final location
        return [(x**2 + y**2)]

    distances = []
    nWalks = 0
    poss_moves = [  (x,y+1),
                    (x+1,y),
                    (x,y-1),
                    (x-1,y)]
    for move in poss_moves:
        if move not in visited:
            distances += recursiveHelper(level-1,visited + [move])

    return distances
