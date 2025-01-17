def map_rb_to_action(pre_rb_mmtc, pre_rb_urllc, pre_rb_embb, rb_mmtc, rb_urllc, rb_embb):
    """
    calculate the action:
    0: keep the current RB configuration
    1: mmtc -> urllc
    2: mmtc -> embb
    3: urllc -> mmtc
    4: urllc -> embb
    5: embb -> mmtc
    6: embb -> urllc
    None: invalid sample
    """
    action = None
    if pre_rb_mmtc == rb_mmtc and pre_rb_urllc == rb_urllc and pre_rb_embb == rb_embb:
        action = 0
    elif pre_rb_mmtc - rb_mmtc == 1 and rb_urllc - pre_rb_urllc == 1 and pre_rb_embb == rb_embb:
        action = 1
    elif pre_rb_mmtc - rb_mmtc == 1 and pre_rb_urllc == rb_urllc and rb_embb - pre_rb_embb == 1:
        action = 2
    elif rb_mmtc - pre_rb_mmtc == 1 and pre_rb_urllc - rb_urllc == 1 and pre_rb_embb == rb_embb:
        action = 3
    elif pre_rb_mmtc == rb_mmtc and pre_rb_urllc - rb_urllc == 1 and rb_embb - pre_rb_embb == 1:
        action = 4
    elif rb_mmtc - pre_rb_mmtc == 1 and pre_rb_urllc == rb_urllc and pre_rb_embb - rb_embb == 1:
        action = 5
    elif pre_rb_mmtc == rb_mmtc and rb_urllc - pre_rb_urllc == 1 and pre_rb_embb - rb_embb == 1:
        action = 6

    return action


def map_action_to_rb(pre_rb_mmtc, pre_rb_urllc, pre_rb_embb, action):
    """
    calculate the next RB configuration:
    0: keep the current RB configuration
    1: mmtc -> urllc
    2: mmtc -> embb
    3: urllc -> mmtc
    4: urllc -> embb
    5: embb -> mmtc
    6: embb -> urllc
    """
    assert action in range(7), "Error! 'action' should be 0, 1, 2, 3, 4, 5, or 6."
    rb_mmtc = pre_rb_mmtc + 0
    rb_urllc = pre_rb_urllc + 0
    rb_embb = pre_rb_embb + 0
    if action == 1:
        if pre_rb_mmtc >= 2:
            rb_mmtc = pre_rb_mmtc - 1
            rb_urllc = pre_rb_urllc + 1
            rb_embb = pre_rb_embb + 0
    elif action == 2:
        if pre_rb_mmtc >= 2:
            rb_mmtc = pre_rb_mmtc - 1
            rb_urllc = pre_rb_urllc + 0
            rb_embb = pre_rb_embb + 1
    elif action == 3:
        if pre_rb_urllc >= 2:
            rb_mmtc = pre_rb_mmtc + 1
            rb_urllc = pre_rb_urllc - 1
            rb_embb = pre_rb_embb + 0
    elif action == 4:
        if pre_rb_urllc >= 2:
            rb_mmtc = pre_rb_mmtc + 0
            rb_urllc = pre_rb_urllc - 1
            rb_embb = pre_rb_embb + 1
    elif action == 5:
        if pre_rb_embb >= 2:
            rb_mmtc = pre_rb_mmtc + 1
            rb_urllc = pre_rb_urllc + 0
            rb_embb = pre_rb_embb - 1
    elif action == 6:
        if pre_rb_embb >= 2:
            rb_mmtc = pre_rb_mmtc + 0
            rb_urllc = pre_rb_urllc + 1
            rb_embb = pre_rb_embb - 1

    return rb_mmtc, rb_urllc, rb_embb
