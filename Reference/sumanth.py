import numpy as np

team_no = 41  # team number

if (team_no % 3 == 0):
    step_cost = -20
elif team_no % 3 == 1:
    step_cost = -10
else:
    step_cost = -5

bellman = 1e-3
discount = 0.99

state_v = np.zeros( shape=(5, 4, 3) )  # (health,arrows,stamina)

states = [(h, a, s) for h in range( 5 )
          for a in range( 4 )
          for s in range( 3 )]

# setting up constant lists
ACTIONS = ["SHOOT", "DODGE", "RECHARGE"]
stamina = [0, 50, 100]
health = [0, 25, 50, 75, 100]
arrow = [0, 1, 2, 3]


def task_setup(no: int) -> str:
    """

    :param no: Task number
    :return: Task file name
    """
    global step_cost, discount, bellman
    if no == 1:
        task_file = "task_1_trace.txt"

    elif no == 2:
        task_file = "task_2_part_1_trace.txt"
        step_cost = -2.5

    elif no == 3:
        discount = 0.1
        step_cost = -2.5
        task_file = "task_2_part_2_trace.txt"

    elif no == 4:
        step_cost = -2.5
        discount = 0.1
        bellman = 1e-10
        task_file = "task_2_part_3_trace.txt"

    with open( task_file, mode='w' ) as f:  # creates an empty new file for the task
        pass

    return task_file


def Print(itr: int, action_index: np.ndarray, task_file: str, debug: int = 1) -> None:
    """

    :param itr: Current iteration number
    :param action_index: Values for best actions
    :param task_file: File to write to
    :param debug: To print to terminal or to write to file
    """
    assert action_index.size == state_v.size

    if debug == 1:
        print( "iteration={}\n".format( itr ) )
        for h, a, s in states:
            temp = -1
            if h != 0:
                temp = ACTIONS[action_index[h][a][s]]

            print( "({0},{1},{2}):{3}=[{4:.3f}]\n".format( h, a, s, temp, state_v[h][a][s] ) )
        print( "\n\n" )

    else:
        with open( task_file, mode='a' ) as f:
            f.write( "iteration={}\n".format( itr ) )
            for h, a, s in states:
                temp = -1
                if h != 0:
                    temp = ACTIONS[action_index[h][a][s]]

                f.write(
                    "({0},{1},{2}):{3}=[{4:.3f}]\n".format( h, a, s, temp, state_v[h][a][s] ) )
            f.write( "\n\n" )


def Utility(s: int, a: int, h: int, action: int) -> float:
    """

    :param s: Stamina
    :param a: Arrows
    :param h: Health
    :param action: Action
    :return: Weighted utility value
    """

    if action == 0 and s > 0 and a > 0:  # Shoot
        return (0.5 * state_v[h, a - 1, s - 1] + 0.5 * state_v[h - 1, a - 1, s - 1]) if (h - 1 != 0) else (
                    0.5 * state_v[h, a - 1, s - 1] + 0.5 * 10)  # if h-1==0, give reward

    elif action == 1 and s > 0:  # dodge
        temp = 0.8 * state_v[h, min( a + 1, 3 ), 0] + 0.2 * state_v[h, a, 0]

        if s == 2:
            temp = 0.8 * 0.8 * state_v[h, min( a + 1, 3 ), 1] + 0.2 * 0.8 * state_v[h, a, 1] + 0.2 * temp

        return temp

    elif action == 2:
        return 0.2 * state_v[h, a, s] + 0.8 * state_v[h, a, min( 2, s + 1 )]

    return -100000.0  # penalty so that this is not considered in the max


def VI_algorithm(task_no: int) -> None:
    """

    :param task_no: Task number
    """
    global state_v

    task_file = task_setup( task_no )

    Temp_table = np.zeros( shape=(5, 4, 3, 3) )  # 4th dimension is the actions

    for itr in range( 1, 1000 ):
        for h, a, s in states:
            for action in range( 3 ):  # produces 0,1,2 for the corresponding actions
                if h == 0:  # skip terminal states
                    continue

                elif task_no == 2 and action == 0:  # for handling task 2 part 2
                    Temp_table[h, a, s, action] = -0.25 + discount * Utility( s, a, h, action )

                else:  # the usual cases to handle
                    Temp_table[h, a, s, action] = step_cost + discount * Utility( s, a, h, action )

        max_table = np.max( Temp_table, axis=3 )  # getting the max utility as per actions
        assert max_table.size == state_v.size

        action_index = np.argmax( Temp_table,
                                  axis=3 )  # getting the corresponding index numbers of the actions with max utility

        val = np.max( np.abs( max_table - state_v ) )

        state_v = max_table  # updated value table
        Print( itr, action_index, debug=0, task_file=task_file )

        if val < bellman:  # end condition for algorithm
            break


VI_algorithm( 1 )  # call as per the task number

# Task Numbers are
# 1 for task 1
# 2 for task 2 part 1
# 3 for task 2 part 2
# 4 for task 2 part 3
