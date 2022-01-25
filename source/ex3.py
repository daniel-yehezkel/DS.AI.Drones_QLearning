import random
import ast
import math

ids = ["0", "0"]


def distance(a, b):
    """The distance between two (x, y) points."""
    xA, yA = a
    xB, yB = b
    return abs(xA - xB) + abs(yA - yB)


def go_to_loc(src, dest):
    moves_to_consider = []
    if src[0] < dest[0]:
        moves_to_consider.append("move_down")

    if src[0] > dest[0]:
        moves_to_consider.append("move_up")

    if src[1] < dest[1]:
        moves_to_consider.append("move_right")

    if src[1] > dest[1]:
        moves_to_consider.append("move_left")

    if len(moves_to_consider) > 0:
        return random.choice(moves_to_consider)
    return None


def finish(obs0):
    if len(obs0["packages"]) == 0:
        return True
    return False


def valuable_actions(obs0):
    if obs0["drone_location"] == obs0["target_location"]:
        for package_name, package_loc in obs0["packages"]:
            if package_loc == "drone":
                return "deliver"

    num_packages = 0
    above_package = False
    for package_name, package_loc in obs0["packages"]:
        if package_loc == "drone":
            num_packages += 1
        if package_loc == obs0["drone_location"]:
            above_package = True

    if num_packages < 2 and above_package:
        return "pick"

    return None


class DroneAgent:
    def __init__(self, n, m, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.mode = 'train'  # do not change this!
        self.n = n
        self.m = m
        self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'wait', 'pick', 'deliver']
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha  # learning constant
        self.gamma = gamma  # discount constant
        self.T_up = math.exp(1)
        self.T_down = 200000 * 30 + 1

    def is_valid(self, drone_loc, move_action):
        if move_action == "move_up":
            if drone_loc[0] + 1 >= self.n:
                return False
            return True
        elif move_action == "move_down":
            if drone_loc[0] - 1 < 0:
                return False
            return True
        elif move_action == "move_right":
            if drone_loc[1] + 1 >= self.m:
                return False
            return True
        else:
            if drone_loc[1] - 1 < 0:
                return False
            return True

    def valid_moves(self, drone_loc):
        valid_moves = [move for move in ['move_up', 'move_down', 'move_left', 'move_right'] if
                       self.is_valid(drone_loc, move)]
        return valid_moves

    def move_by_heuristic(self, drone_loc, dest_loc):
        if random.random() < 0.3:
            return random.choice(self.valid_moves(drone_loc))
        next_move = go_to_loc(drone_loc, dest_loc)
        if next_move is not None:
            return next_move
        return None

    def select_action(self, obs0):
        obs0["packages"] = sorted(list(obs0["packages"]), key=lambda x: x[0])
        # print(obs0)
        if self.mode == 'train':
            self.T_up += .1
            self.T_down -= 1
            drone_loc = obs0["drone_location"]
            if random.random() < 0.7:
                if random.random() < 1 / 2:  # self.T_up:
                    if finish(obs0):
                        return "reset"
                    valuable_actions_temp = valuable_actions(obs0)
                    if valuable_actions_temp is not None:
                        return valuable_actions_temp
                    if not finish(obs0):
                        target_loc = obs0["target_location"]
                        num_packages = 0
                        for package_name, package_loc in obs0["packages"]:
                            if package_loc == "drone":
                                num_packages += 1
                        if num_packages == 2 or (num_packages == len(obs0["packages"])):
                            next_move = self.move_by_heuristic(drone_loc, target_loc)
                            if next_move is not None:
                                return next_move

                        if random.random() < 0.7:
                            sorted_packages = sorted(
                                [(package_loc, distance(package_loc, drone_loc)) for
                                 _, package_loc in
                                 obs0["packages"] if package_loc != "drone"], key=lambda x: x[1])
                            if len(sorted_packages) > 0:
                                closest_package_loc = sorted_packages[0][0]
                                next_move = self.move_by_heuristic(drone_loc, closest_package_loc)
                                if next_move is not None:
                                    return next_move

                                all_actions = set(self.actions)
                                for move_action in ['move_up', 'move_down', 'move_left', 'move_right']:
                                    if not self.is_valid(drone_loc, move_action) and drone_loc != 'random':
                                        all_actions = all_actions.difference({move_action})

                                return random.choice(list(all_actions.difference({"pick", "deliver"})))

                    return random.choice(list(self.actions))

            else:
                obs0_r = repr(obs0)
                Z = sum([math.exp(self.get_q(obs0_r, a) / self.T_down) for a in self.actions])
                probs = [math.exp(self.get_q(obs0_r, a) / self.T_down) / Z for a in self.actions]
                return random.choices(self.actions, probs)[0]

        # Eval
        if finish(obs0):
            return "reset"

        valuable_actions_temp = valuable_actions(obs0)
        if valuable_actions_temp is not None:
            return valuable_actions_temp
        obs0_r = repr(obs0)
        q = [self.get_q(obs0_r, a) for a in self.actions]
        maxQ = max(q)
        count = q.count(maxQ)
        # In case there are several state-action max values we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)
        return self.actions[i]

    def train(self):
        self.mode = 'train'  # do not change this!

    def eval(self):
        self.mode = 'eval'  # do not change this!

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def update(self, obs0, action, obs1, reward):
        obs0["packages"] = sorted(list(obs0["packages"]), key=lambda x: x[0])
        obs1["packages"] = sorted(list(obs1["packages"]), key=lambda x: x[0])
        obs0_r = repr(obs0)
        obs1_r = repr(obs1)

        q_max = max([self.get_q(obs1_r, a) for a in self.actions])
        old_q = self.q.get((obs0_r, action), None)
        if old_q is None:
            self.q[(obs0_r, action)] = reward
        else:
            self.q[(obs0_r, action)] = old_q + self.alpha * (reward + self.gamma * q_max - old_q)
