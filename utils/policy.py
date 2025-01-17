import numpy as np


def policy(num_users_mmtc, num_users_urllc, num_users_embb, num_rb_mmtc, num_rb_urllc, num_rb_embb):
    """
    This function returns a RB configuration for the next step given the current RB configuration.
        Args:
            num_users_mmtc: the number of users in mmtc
            num_users_urllc: the number of users in urllc
            num_users_embb: the number of users in embb
            num_rb_mmtc: the number of RBs for mmtc in the current step
            num_rb_urllc: the number of RBs for urllc in the current step
            num_rb_embb: the number of RBs for embb in the current step
        Returns:
            num_rb_mmtc_next: the number of RBs for mmtc in the next step
            num_rb_urllc_next: the number of RBs for urllc in the next step
            num_rb_embb_next: the number of RBs for embb in the next step


    """
    total_rb = 18
    assert num_rb_mmtc + num_rb_urllc + num_rb_embb == total_rb, "The total number of RB of the input should be 18."

    num_rb_mmtc_next = num_rb_mmtc + 0
    num_rb_urllc_next = num_rb_urllc + 0
    num_rb_embb_next = num_rb_embb + 0

    # random change of RB configuration
    action = np.random.choice(7)
    """
    0: keep the current RB configuration
    1: mmtc -> urllc
    2: mmtc -> embb
    3: urllc -> mmtc
    4: urllc -> embb
    5: embb -> mmtc
    6: embb -> urllc
    """
    if action == 1:
        if num_rb_mmtc >= 2:
            num_rb_mmtc_next = num_rb_mmtc - 1
            num_rb_urllc_next = num_rb_urllc + 1
            num_rb_embb_next = num_rb_embb + 0
    elif action == 2:
        if num_rb_mmtc >= 2:
            num_rb_mmtc_next = num_rb_mmtc - 1
            num_rb_urllc_next = num_rb_urllc + 0
            num_rb_embb_next = num_rb_embb + 1
    elif action == 3:
        if num_rb_urllc >= 2:
            num_rb_mmtc_next = num_rb_mmtc + 1
            num_rb_urllc_next = num_rb_urllc - 1
            num_rb_embb_next = num_rb_embb + 0
    elif action == 4:
        if num_rb_urllc >= 2:
            num_rb_mmtc_next = num_rb_mmtc + 0
            num_rb_urllc_next = num_rb_urllc - 1
            num_rb_embb_next = num_rb_embb + 1
    elif action == 5:
        if num_rb_embb >= 2:
            num_rb_mmtc_next = num_rb_mmtc + 1
            num_rb_urllc_next = num_rb_urllc + 0
            num_rb_embb_next = num_rb_embb - 1
    elif action == 6:
        if num_rb_embb >= 2:
            num_rb_mmtc_next = num_rb_mmtc + 0
            num_rb_urllc_next = num_rb_urllc + 1
            num_rb_embb_next = num_rb_embb - 1

    assert num_rb_mmtc_next + num_rb_urllc_next + num_rb_embb_next == total_rb, "The total number of RB of the input should be 18."
    assert num_rb_mmtc_next >= 1, "The number of RB for mmtc in the next step should be greater than or equal to 1"
    assert num_rb_urllc_next >= 1, "The number of RB for urllc in the next step should be greater than or equal to 1"
    assert num_rb_embb_next >= 1, "The number of RB for embb in the next step should be greater than or equal to 1"

    return num_rb_mmtc_next, num_rb_urllc_next, num_rb_embb_next


"""
test in the following
"""
# def main():
#     num_rb_mmtc, num_rb_urllc, num_rb_embb = 6, 6, 6
#     for i in range(10000):
#         print(num_rb_mmtc, num_rb_urllc, num_rb_embb)
#         num_rb_mmtc_next, num_rb_urllc_next, num_rb_embb_next = policy(3, 6, 1, num_rb_mmtc, num_rb_urllc, num_rb_embb)
#         num_rb_mmtc, num_rb_urllc, num_rb_embb = num_rb_mmtc_next, num_rb_urllc_next, num_rb_embb_next
#
#
# if __name__ == "__main__":
#     main()
