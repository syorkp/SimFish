import h5py
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.signal import detrend
from scipy.stats import zscore

root = tk.Tk()
root.withdraw()

#file_path = filedialog.askopenfilename()

# base_path = './Training-Output/prey_and_predator2_prev-4/episodes/'
# episodes = ['Episode 628.h5', 'Episode 630.h5', 'Episode 632.h5', 'Episode 634.h5', 'Episode 636.h5',
#             'Episode 638.h5', 'Episode 640.h5', 'Episode 642.h5', 'Episode 644.h5', 'Episode 646.h5',
#             'Episode 648.h5', 'Episode 650.h5', 'Episode 652.h5', 'Episode 654.h5', 'Episode 656.h5',
#             'Episode 658.h5', 'Episode 660.h5', 'Episode 662.h5', 'Episode 664.h5', 'Episode 666.h5',
#             'Episode 668.h5', 'Episode 670.h5', 'Episode 672.h5', 'Episode 674.h5', 'Episode 676.h5',
#             'Episode 678.h5', 'Episode 680.h5', 'Episode 682.h5', 'Episode 684.h5', 'Episode 686.h5',
#             'Episode 688.h5', 'Episode 690.h5', 'Episode 692.h5', 'Episode 694.h5', 'Episode 696.h5',
#             'Episode 698.h5', 'Episode 700.h5', 'Episode 702.h5', 'Episode 704.h5', 'Episode 706.h5',
#             'Episode 708.h5', 'Episode 710.h5', 'Episode 712.h5', 'Episode 714.h5', 'Episode 716.h5',
#             'Episode 718.h5', 'Episode 720.h5', 'Episode 722.h5', 'Episode 724.h5', 'Episode 726.h5',
#             'Episode 728.h5', 'Episode 730.h5', 'Episode 732.h5', 'Episode 734.h5', 'Episode 736.h5',
#             'Episode 738.h5', 'Episode 740.h5', 'Episode 742.h5', 'Episode 744.h5', 'Episode 746.h5',
#             'Episode 748.h5', 'Episode 750.h5', 'Episode 752.h5', 'Episode 754.h5', 'Episode 756.h5',
#             'Episode 758.h5', 'Episode 760.h5', 'Episode 762.h5']

#base_path = './Training-Output/prey_basic_static-10/episodes/'
base_path = '/home/asaph/RL-Project/SimFish/Training-Output/dqn_basic_f-2/episodes/'
episodes = ['Episode 280.h5']

# base_path = './Training-Output/prey_and_predator2_prev-1/episodes/'
# episodes = ['Episode 630.h5', 'Episode 632.h5', 'Episode 634.h5', 'Episode 636.h5',
#             'Episode 638.h5', 'Episode 640.h5', 'Episode 642.h5', 'Episode 644.h5', 'Episode 646.h5',
#             'Episode 648.h5', 'Episode 650.h5']

ep = 0
left_eye_right_advantages = np.zeros((len(episodes), 139))
left_eye_left_advantages = np.zeros((len(episodes), 139))
right_eye_right_advantages = np.zeros((len(episodes), 139))
right_eye_left_advantages = np.zeros((len(episodes), 139))

for episode in episodes:
    file_path = base_path + episode
    with h5py.File(file_path, "r") as file:
        group = list(file.keys())[0]
        rnn = file[group]['rnn_state_actor'][:, 1, :]
        rnn_ref = file[group]['rnn_state_actor_ref'][:, 1, :]
        #adv = np.array(file[group]['advantage'])[:, 0, :]
        #adv_ref = np.array(file[group]['advantage_ref'])[:, 0, :]
        #value = np.array(file[group]['value'])
        #value_ref = np.array(file[group]['value_ref'])
        obs = np.array(file[group]['observation'])

    right_advantage_normal = adv[:, 1] + adv[:, 4]# + adv[:, 7]# + adv[:, 10]
    right_advantage_ref = adv_ref[:, 1] + adv_ref[:, 4]# + adv_ref[:, 7]# + adv_ref[:, 10]
    left_advantage_normal = adv[:, 2] + adv[:, 5]# + adv[:, 8]# + adv[:, 11]
    left_advantage_ref = adv_ref[:, 2] + adv_ref[:, 5]# + adv_ref[:, 8]# + adv_ref[:, 11]

    total_right_advantage = (right_advantage_normal + right_advantage_ref) / 2
    total_left_advantage = (left_advantage_normal + left_advantage_ref) / 2

    r_vs_l = total_right_advantage - total_left_advantage

    left_advantage_steps = np.where(r_vs_l < -1)[0]
    right_advantage_steps = np.where(r_vs_l > 1)[0]

    left_eye_right_advantage = np.zeros((len(right_advantage_steps), obs.shape[1]))
    left_eye_left_advantage = np.zeros((len(left_advantage_steps), obs.shape[1]))
    right_eye_right_advantage = np.zeros((len(right_advantage_steps), obs.shape[1]))
    right_eye_left_advantage = np.zeros((len(left_advantage_steps), obs.shape[1]))

    for step in range(len(left_advantage_steps)):
        if left_advantage_steps[step] < 2 or left_advantage_steps[step] > len(obs) - 1:
            continue
        hist = obs[left_advantage_steps[step]-2:left_advantage_steps[step]+1, :, 1, :]
        left_eye_left_advantage[step, :] = np.mean(hist[:, :, 0], axis=0)
        right_eye_left_advantage[step, :] = np.mean(hist[:, :, 1], axis=0)

    for step in range(len(right_advantage_steps)):
        if right_advantage_steps[step] < 2 or right_advantage_steps[step] > len(obs) - 1:
            continue
        hist = obs[right_advantage_steps[step]-2:right_advantage_steps[step]+1, :, 1, :]
        left_eye_right_advantage[step, :] = np.mean(hist[:, :, 0], axis=0)
        right_eye_right_advantage[step, :] = np.mean(hist[:, :, 1], axis=0)
    
    left_eye_right_advantages[ep, :] = np.mean(left_eye_right_advantage, axis=0)
    left_eye_left_advantages[ep, :] = np.mean(left_eye_left_advantage, axis=0)
    right_eye_right_advantages[ep, :] = np.mean(right_eye_right_advantage, axis=0)
    right_eye_left_advantages[ep, :] = np.mean(right_eye_left_advantage, axis=0)

    ep += 1

        



plt.figure()
plt.subplot(2, 2, 1)
plt.plot(np.mean(left_eye_left_advantages, axis=0))
plt.ylim(20, 45)
plt.title('left eye left advantage')
plt.subplot(2, 2, 2)
plt.plot(np.mean(right_eye_left_advantages, axis=0))
plt.ylim(20, 45)
plt.title('right eye left advantage')
plt.subplot(2, 2, 3)
plt.plot(np.mean(left_eye_right_advantages, axis=0))
plt.ylim(20, 45)
plt.title('left eye right advantage')
plt.subplot(2, 2, 4)
plt.plot(np.mean(right_eye_right_advantages, axis=0))
plt.ylim(20, 45)
plt.title('right eye right advantage')
plt.show()