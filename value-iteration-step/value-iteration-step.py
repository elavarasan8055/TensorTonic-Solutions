import numpy as np
def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    values_np=np.array(values)
    transitions_np=np.array(transitions)
    rewards_np=np.array(rewards)
    
    states=rewards_np.shape[0]
    actions=rewards_np.shape[1]

    return_value= [float(np.max([rewards_np[state,action]+gamma*np.dot(transitions_np[state,action,:],values_np) for action in range(actions)])) for state in range(states)]
    return return_value    


