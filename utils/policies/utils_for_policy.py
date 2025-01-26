import torch


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


def valid_actions(pre_rb_mmtc, pre_rb_urllc, pre_rb_embb):
    """
        Return a list of all valid actions
        action index:
        0: keep the current RB configuration
        1: mmtc -> urllc
        2: mmtc -> embb
        3: urllc -> mmtc
        4: urllc -> embb
        5: embb -> mmtc
        6: embb -> urllc
    """
    valid_actions_list = []
    for action in range(7):
        is_valid = False
        if action == 0:
            is_valid = True
        elif action == 1 or action == 2:
            if pre_rb_mmtc >= 2:
                is_valid = True
        elif action == 3 or action == 4:
            if pre_rb_urllc >= 2:
                is_valid = True
        elif action == 5 or action == 6:
            if pre_rb_embb >= 2:
                is_valid = True
        if is_valid:
            valid_actions_list.append(action)

    return valid_actions_list


def valid_actions_batch_tensor_version(pre_rb_mmtc, pre_rb_urllc, pre_rb_embb):
    """
        pre_rb_mmtc: (N)
        pre_rb_urllc: (N)
        pre_rb_embb: (N)
        Return a batch of masks that mask the invalid actions
        action index:
        0: keep the current RB configuration
        1: mmtc -> urllc
        2: mmtc -> embb
        3: urllc -> mmtc
        4: urllc -> embb
        5: embb -> mmtc
        6: embb -> urllc
    """
    is_valid = torch.zeros((pre_rb_mmtc.shape[0], 7), dtype=torch.bool)
    for action in range(7):
        if action == 0:
            is_valid[:, action].fill_(True)
        elif action == 1 or action == 2:
            is_valid[pre_rb_mmtc >= 2, action] = True
        elif action == 3 or action == 4:
            is_valid[pre_rb_urllc >= 2, action] = True
        elif action == 5 or action == 6:
            is_valid[pre_rb_embb >= 2, action] = True

    return is_valid
