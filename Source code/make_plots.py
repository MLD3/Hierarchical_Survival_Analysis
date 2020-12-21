'''
make the bar plots seen in the paper
'''

import numpy as np
import matplotlib.pyplot as plt

proposed_name = 'Proposed'

###################################################################################################
'''
make a bar plot given the data, for datasets with competing or semi-competing events
'''
def make_bar_plot(combined_data, ybounds, methods, axis_labs, \
                   xtick_labs, colors):
    num_methods = len(combined_data)
    _, ax1 = plt.subplots()
    
    for i in range(num_methods):
        for j in range(combined_data[i].shape[1]):
            data = combined_data[i]
            
            error_bars = data[[1, 2], :]
            error_bars[0, :] = data[0, :] - error_bars[0, :]
            error_bars[1, :] = error_bars[1, :] - data[0, :]
            
            xpos = (np.arange(0, (num_methods + 1) * data.shape[1], num_methods + 1) + i) 
            if j == 0:
                ax1.bar(xpos[j], data[0, j], yerr=error_bars[:, j].reshape(-1, 1), capsize=2, align='center', alpha=0.5, label=methods[i], color=colors[i])
            else:
                ax1.bar(xpos[j], data[0, j], yerr=error_bars[:, j].reshape(-1, 1), capsize=2, align='center', alpha=0.5, color=colors[i])

    plt.xticks([], [])
    if xtick_labs is None:
        xtick_scale = np.arange(combined_data[0].shape[1])    
        plt.xticks(xtick_scale * (num_methods + 1) + ((num_methods - 1) / 2), xtick_scale + 1)    
    else:
        plt.xticks(xtick_labs[0], xtick_labs[1]) 
        
    #ax1.legend(bbox_to_anchor=(0, 1.15), ncol=1, loc='upper left')  
    ax1.set_ylabel(axis_labs[0])
    ax1.yaxis.grid() 
    ax1.set_ylim(ybounds[0])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.show()


###################################################################################################
'''
call the make bar plot function for synthetic results (ignore third column)
'''
def make_synth_plots(): 
    global proposed_name 
    
    oracle = np.array([[0.831,  0.836, 0.843], \
         [0.816, 0.821, 0.814], \
         [0.846, 0.850, 0.872]])
    
    indep = np.array([[0.562, 0.547, 0.578], \
         [0.540, 0.528, 0.454], \
         [0.584, 0.565, 0.611]])
    deephit = np.array([[0.612, 0.585, 0.607], \
         [0.590, 0.567, 0.576], \
         [0.633, 0.604, 0.639]])
    proposed = np.array([[0.776,  0.776, 0.832], \
         [0.760, 0.761, 0.802], \
         [0.791, 0.791, 0.862]])
    
    colors = ['C0', 'C1', 'C9', 'C3']
    methods = ['Independent', 'DeepHit', proposed_name, 'Oracle']
    make_bar_plot([indep, deephit, proposed, oracle], [(0.4, 0.9)], \
                  methods, ['Performance', 'Metric'], \
                  [[1.5, 6.5, 11.5], ['C-Index', 'Global\nConsistency', 'Local\nConsistency']], colors)
    
      
'''
call the make bar plot function for adni results
'''    
def make_adni_bar_plots():
    global proposed_name 
    
    indep = np.array([[0.751, 0.727, 0.944], \
         [0.703, 0.699, 0.910], \
         [0.800, 0.763, 0.971]])
    deephit = np.array([[0.824, 0.804, 0.952], \
         [0.749, 0.729, 0.921], \
         [0.889, 0.882, 0.976]])    
    proposed = np.array([[0.911,  0.906, 0.971], \
         [0.888, 0.884, 0.946], \
         [0.931, 0.926, 0.984]]) 
    
    colors = ['C0', 'C1', 'C9', 'C3']
    methods = ['Independent', 'DeepHit', proposed_name]
    make_bar_plot([indep, deephit, proposed], [(0.5, 1)], methods, \
                   ['Performance', 'Metric'], [[1, 5, 9], ['C-Index', 'Global\nConsistency', 'Local\nConsistency']], colors)

        
'''
call the make bar plot function for mimic results
'''    
def make_mimic_bar_plots():
    global proposed_name
    
    indep = np.array([[0.554, 0.560, 0.637], \
         [0.532, 0.544, 0.616], \
         [0.575, 0.577, 0.658]])
    deephit = np.array([[0.608, 0.596, 0.655], \
         [0.582, 0.547, 0.633], \
         [0.637, 0.583, 0.678]])
    proposed = np.array([[0.680, 0.680, 0.751], \
         [0.653, 0.653, 0.713], \
         [0.706, 0.706, 0.787]])
    
    colors = ['C0', 'C1', 'C9', 'C3']
    methods = ['Independent', 'DeepHit', proposed_name]
    make_bar_plot([indep, deephit, proposed], [(0.45, 0.82)], methods, \
                   ['Performance', 'Metric'], [[1, 5, 9], ['C-Index', 'Global\nConsistency', 'Local\nConsistency']], colors)
    
    
'''
call the make bar plot function for seer results
'''    
def make_seer_bar_plots():
    global proposed_name
    
    indep = np.array([[0.789, 0.781, 0.740], \
         [0.766, 0.759, 0.693], \
         [0.810, 0.802, 0.787]])
    deephit = np.array([[0.799, 0.790, 0.778], \
         [0.775, 0.767, 0.734], \
         [0.820, 0.811, 0.819]])
    proposed = np.array([[0.808, 0.801, 0.781], \
         [0.784, 0.777, 0.743], \
         [0.830, 0.823, 0.822]])
    
    colors = ['C0', 'C1', 'C9', 'C3']
    methods = ['Independent', 'DeepHit', proposed_name]
    make_bar_plot([indep, deephit, proposed], [(0.5, 0.85)], methods, \
                   ['Performance', 'Metric'], [[1, 5, 9], ['C-Index', 'Global\nConsistency', 'Local\nConsistency']], colors)

    
###################################################################################################
if __name__ == '__main__':
    make_synth_plots()
    make_adni_bar_plots()
    make_mimic_bar_plots()
    make_seer_bar_plots()
    