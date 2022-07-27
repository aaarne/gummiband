import numpy as np
from collections import deque
from itertools import islice


def create_more_points_on_convergence_schedule(
        n_points,
        initial_rate,
        increase_points_vel=0.01,
        final_vel=2e-3,
        n_max=250,
        # No Progress Checker
        queue_len=40,
        recent_len=10,
        required_improvement_ratio=0.99,
        minimum_com_vel=0.01,
):
    n = n_points
    lr = initial_rate

    vel_queue = deque(maxlen=queue_len)

    def s(e, l, vel, c, com_vel):
        nonlocal n, lr

        vel_queue.append(vel)
        if e > 10:
            if vel < increase_points_vel and n < n_max:
                lr = initial_rate
                n += 10
                vel_queue.clear()
            if n >= n_max and vel < final_vel:
                raise StopIteration
            if len(vel_queue) == queue_len:
                average = np.mean(vel_queue)
                recent_average = np.mean(np.array([*islice(vel_queue, queue_len - recent_len, queue_len)]))
                if recent_average >= required_improvement_ratio * average:
                    if com_vel < minimum_com_vel:
                        print(f"Lowering time step from {lr} to {lr / 2}.")
                        vel_queue.clear()
                        lr /= 2
                    else:
                        print(f"No speed progress, but com movement.")

        return lr, n, n

    return s
