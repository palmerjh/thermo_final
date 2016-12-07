import random
import numpy as np

from collections import deque, namedtuple
Monomer = namedtuple("Monomer", "index v_in v_out turn")   # simple container used to hold Monomer data: index, vector_in, vector_out, turn (cross_product)

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
                STOP    : np.array([0,0]) }

    reverse_deltas = {  (0,1)   : UP,
                        (1,0)   : RIGHT,
                        (0,1)   : DOWN,
                        (-1,0)  : LEFT,
                        (0,0)   : STOP }

    # left turn --> +1
    # right turn --> -1
    # parallel/anti-parallel --> 0
    cross_product = {   UP      : { UP      : 0,
                                    RIGHT   : -1,
                                    DOWN    : 0,
                                    LEFT    : 1,
                                    STOP    : 0 },
                        RIGHT   : { UP      : 1,
                                    RIGHT   : 0,
                                    DOWN    : -1,
                                    LEFT    : 0,
                                    STOP    : 0 },
                        DOWN    : { UP      : 0,
                                    RIGHT   : 1,
                                    DOWN    : 0,
                                    LEFT    : -1,
                                    STOP    : 0 },
                        LEFT    : { UP      : -1,
                                    RIGHT   : 0,
                                    DOWN    : 1,
                                    LEFT    : 0,
                                    STOP    : 0 },
                        STOP    : { UP      : 0,
                                    RIGHT   : 0,
                                    DOWN    : 0,
                                    LEFT    : 0,
                                    STOP    : 0 } }

    # left turn --> 0
    # right turn --> 0
    # parallel --> +1
    # anti-parallel --> -1
    dot_product = {     UP      : { UP      : 1,
                                    RIGHT   : 0,
                                    DOWN    : -1,
                                    LEFT    : 0,
                                    STOP    : 0 },
                        RIGHT   : { UP      : 0,
                                    RIGHT   : 1,
                                    DOWN    : 0,
                                    LEFT    : -1,
                                    STOP    : 0 },
                        DOWN    : { UP      : -1,
                                    RIGHT   : 0,
                                    DOWN    : 1,
                                    LEFT    : 0,
                                    STOP    : 0 },
                        LEFT    : { UP      : 0,
                                    RIGHT   : -1,
                                    DOWN    : 0,
                                    LEFT    : 1,
                                    STOP    : 0 },
                        STOP    : { UP      : 0,
                                    RIGHT   : 0,
                                    DOWN    : 0,
                                    LEFT    : 0,
                                    STOP    : 0 } }

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
        #start_mon = Monomer(0, Moves.STOP, Moves.STOP, 0))
        self.pos = np.array(start)

        self.probs = {  Moves.UP    : up,
                        Moves.RIGHT : right,
                        Moves.DOWN  : down,
                        Moves.LEFT  : left  }

        self.nSteps = 0
        self.visited = {}

        self.prev_move = Moves.STOP
        self.turns = []
        self.can_walk = True

    def walk(self):
        if self.can_walk:
            rando = random.random()
            possible_moves = [move for move in Moves.moves if self.test_move(move)]

            renormalized_probs = self.renormalize(possible_moves)

            prob_threshold = 0
            for move in possible_moves:
                prob_threshold += renormalized_probs[move]
                if rando < prob_threshold:
                    return self.apply_move(move)

            # some other error occured
            self.can_walk = False
            return -1


    def apply_move(self,move,test=False):
        new_xy = self.pos + Moves.deltas[move]

        '''
        print x,y
        print new_xy
        print self.visited

        raw_input('press enter...')
        '''

        if not test:
            cur_mon = createMonomer(self.nSteps,self.prev_move,move)
            self.turns.append(cur_mon.turn)
            self.visited.update({tuple(self.pos)    : cur_mon})

            self.prev_move = move
            self.pos = new_xy
            self.nSteps += 1

        return 0

    # tests move to see if it can be done (i.e. leads to unvisited square)
    # if so, tests to see if SAW would be trapped in new position
    def test_move(self,move):
        new_xy = self.pos + Moves.deltas[move]

        # definitely can't move here if new_xy in self.visited
        if tuple(new_xy) in self.visited.keys():
            return False

        # now time for trap testing...

        cur_mon = createMonomer(self.nSteps,self.prev_move,move)

        new_poss_moves = Moves.moves[:]
        #print new_poss_moves
        #print move
        Moves.reverse[move]
        # know we can't move backwards
        new_poss_moves.remove(Moves.reverse[move])
        # excludes prev_pos because that is a given
        new_blocked = []

        for new_move in new_poss_moves:
            new_new_xy = new_xy + Moves.deltas[new_move]
            if tuple(new_new_xy) in self.visited.keys():
                new_poss_moves.remove(new_move)
                new_blocked.append((tuple(new_new_xy), new_move))

        # definitely trapped if every possible next_move is closed
        if len(new_blocked) == 3:
            return False

        # definitely not trapped if every possible next_move is open
        if len(new_blocked) == 0:
            return True

        # possibly trapped
        if len(new_blocked) == 1:
            new_new_xy, new_move = new_blocked[0]
            dot = Moves.dot_product[move][new_move]

            # head on collision with singular previously visited site (i.e. T-bone)
            # can always go one direction or the other --> definitely not trapped
            if dot == 1:
                return True
            # else blocked/visited square on either side of new_xy and more analysis is needed to determine if trap

        if len(new_blocked) == 2:
            dots = [Moves.dot_product[move][new_move] for (new_new_xy, new_move) in new_blocked]
            # at least one of the new_blocked is head-on --> SAW would be in corner
            # can prove that a SAW in corner is never trapped
            if sum(dots) == 1:
                return True

            # else sum(dots) == 0 --> we have two new_blocked squares on either side of new_xy

        # at this point we know we must find loops to determine if move caused
        # walker to become trapped
        # Possibilities:
        #   - one blocked/visited square on either side of new_xy
        #   - two blocked/visited squares on either side of new_xy

        # possibly trapped
        # one blocked/visited square on either side of new_xy
        if len(new_blocked) == 1:
            new_new_xy, new_move = new_blocked[0]
            blocked_mon = self.visited[tuple(new_new_xy)]

            new_mon = createMonomer(self.nSteps+1, move, new_move)

            # temporarily update turns and visited in order to find loops
            # and decide whether all next_next_moves would place SAW inside loop
            temp_turns = self.turns + [cur_mon.turn,new_mon.turn]
            temp_visited = self.visited.copy()

            temp_visited.update({tuple(self.pos)    : cur_mon})
            temp_visited.update({tuple(new_xy)      : new_mon})

            # should replace vector_in of this blocked square with new_move to create full loop
            # keep vector_out of this blocked square the same
            # should update the turn entry for new_new_xy indexed by index
            index = blocked_mon.index
            new_new_mon = createMonomer(index,new_move,blocked_mon.v_out)
            temp_visited.update({tuple(new_new_xy)      : new_new_mon})
            temp_turns[index] = new_new_mon.turn

            # know this is true because clockwise loop has more right turns than left turns
            # and according to Moves.cross_product, right turns have a value of -1
            loop_isClockwise = sum(temp_turns[index:]) < 0

            # either a left (+1) or right (-1) turn
            fate_determining_turn = new_mon.turn

            # if the loop is clockwise:
            #   -if loop was completed using a right (-1) turn --> not trapped
            #   -else --> trapped
            if loop_isClockwise:
                return (fate_determining_turn == -1)
            # else the loop is anti-clockwise:
            #   -if loop was completed using a left (+1) turn --> not trapped
            #   -else --> trapped
            return (fate_determining_turn == 1)

        # now we know that there are two blocked/visited squares on either side of new_xy
        # there are two subcases to consider
        #   -1) orientations of two blocked/visited squares are equal and anti-parallel to move
        #       -in this case, new_xy creates TWO loops that must both be considered/analyzed
        #   -2) orientations of two blocked/visited squares are opposite
        #       -in this case, v_in comes from square with orientation anti-parallel to that of new_xy
        #       -v_out goes towards square with orientation parallel to that of new_xy
        #       -only one loop is created
        #   -Note: impossible for orientations of two blocked/visited squares to be equal and parallel to move
        #### UPDATE: realized that I can treat both subcases in the same way with two loops

        # fetches the monomer for each new_blocked square
        blocked_mons = [self.visited[tuple(new_new_xy)] for (new_new_xy, new_move) in new_blocked]
        for i in range(2):
            new_new_xy, new_move = new_blocked[i]
            blocked_mon = blocked_mons[i]

            new_mon = createMonomer(self.nSteps+1, move, new_move)

            # temporarily update turns and visited in order to find loops
            # and decide whether all next_next_moves would place SAW inside loop
            temp_turns = self.turns + [cur_mon.turn,new_mon.turn]
            temp_visited = self.visited.copy()

            temp_visited.update({tuple(self.pos)    : cur_mon})
            temp_visited.update({tuple(new_xy)      : new_mon})

            # should replace vector_in of this blocked square with new_move to create full loop
            # keep vector_out of this blocked square the same
            # should update the turn entry for new_new_xy indexed by index
            index = blocked_mon.index
            new_new_mon = createMonomer(index,new_move,blocked_mon.v_out)
            temp_visited.update({tuple(new_new_xy)      : new_new_mon})
            temp_turns[index] = new_new_mon.turn

            # know this is true because clockwise loop has more right turns than left turns
            # and according to Moves.cross_product, right turns have a value of -1
            loop_isClockwise = sum(temp_turns[index:]) < 0

            # either a left (+1) or right (-1) turn
            fate_determining_turn = new_mon.turn

            # if the loop is clockwise:
            #   -if loop was completed using a right (-1) turn --> not trapped
            #   -else --> trapped
            if loop_isClockwise:
                trapped = not (fate_determining_turn == -1)
            # else the loop is anti-clockwise:
            #   -if loop was completed using a left (+1) turn --> not trapped
            #   -else --> trapped
            else:
                trapped = not (fate_determining_turn == 1)

            # we can short-cicuit the OR (just need one of the loops to trap SAW to return False)
            if trapped:
                return False

        # neither of the loops trapped the SAW
        return True


        '''
        # fetches the monomer for each new_blocked square
        blocked_mons = [self.visited[tuple(new_new_xy)] for (new_new_xy, new_move) in new_blocked]
        parallel_vectors = []
        for b_mon in blocked_mons:
            # vector_in of this blocked square lies parallel to move
            if Moves.cross_product[move][b_mon.v_in] == 0:
                parallel_vectors.append(b_mon.v_in)
            # by process-of-elimination, vector_out of this blocked square must lie parallel to move
            else:
                parallel_vectors.append(b_mon.v_out)

        # subcase 1 as described above
        if parallel_vectors[0] == parallel_vectors[1]:
            for i in range(2):
                new_new_xy, new_move = new_blocked[i]
                blocked_mon = blocked_mons[i]

                new_mon = createMonomer(self.nSteps+1, move, new_move)

                # temporarily update turns and visited in order to find loops
                # and decide whether all next_next_moves would place SAW inside loop
                temp_turns = self.turns + [cur_mon.turn,new_mon.turn]
                temp_visited = self.visited.copy()

                temp_visited.update({tuple(self.pos)    : cur_mon})
                temp_visited.update({tuple(new_xy)      : new_mon})

                # should replace vector_in of this blocked square with new_move to create full loop
                # keep vector_out of this blocked square the same
                # should update the turn entry for new_new_xy indexed by index
                index = blocked_mon.index
                new_new_mon = createMonomer(index,new_move,blocked_mon.v_out)
                temp_visited.update({tuple(new_new_xy)      : new_new_mon})
                temp_turns[index] = new_new_mon.turn

                # know this is true because clockwise loop has more right turns than left turns
                # and according to Moves.cross_product, right turns have a value of -1
                loop_isClockwise = sum(temp_turns[index:]) < 0

                # either a left (+1) or right (-1) turn
                fate_determining_turn = new_mon.turn

                # if the loop is clockwise:
                #   -if loop was completed using a right (-1) turn --> not trapped
                #   -else --> trapped
                if loop_isClockwise:
                    trapped = not (fate_determining_turn == -1)
                # else the loop is anti-clockwise:
                #   -if loop was completed using a left (+1) turn --> not trapped
                #   -else --> trapped
                else:
                    trapped = not (fate_determining_turn == 1)

                # we can short-cicuit the OR (just need one of the loops to trap SAW to return False)
                if trapped:
                    return False

            # neither of the loops trapped the SAW
            return True

        # subcase 2 as described above
        blocked_mons = [self.visited[tuple(new_new_xy)] for (new_new_xy, new_move) in new_blocked]
        parallel_vectors = []
        for b_mon in blocked_mons:
            # vector_in of this blocked square lies parallel to move
            if Moves.cross_product[move][b_mon.v_in] == 0:
                parallel_vectors.append(b_mon.v_in)
            # by process-of-elimination, vector_out of this blocked square must lie parallel to move
            else:
                parallel_vectors.append(b_mon.v_out)

        '''
    def getR2(self):
        x,y = tuple(self.pos)
        return (x)**2 + (y)**2

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

def createMonomer(index,v_in,v_out):
    turn = Moves.cross_product[v_in][v_out]
    return Monomer(index,v_in,v_out,turn)

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
