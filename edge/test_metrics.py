import numpy as np
import matplotlib.pyplot as plt
import randomname
from edge_experiment import ttc
from edge_experiment import mrel

class Robot:
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.ID = 'Rx_'+randomname.get_name()
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0].item(), pos[1].item()]
        acc = 0.01
    def read(self, dt):
        pos = [0,0]
        self.pos[0] += self.vel[0].item() *dt 
        self.pos[1] += self.vel[1].item() *dt 
        rng = np.random.default_rng()
        facts = rng.normal(0, self.noise_std,2) 

        p0 = self.pos[0] + facts[0]
        p1 = self.pos[1] + facts[1]
        return [p0.item(), p1.item()]

def main():
    #Create positon speed for robots
    pos_r1 = np.random.randint(75, high=120, size=(2))
    vel_r1 = np.random.normal(0, 5, 2)
    pos_r2 = np.random.randint(75, high=120, size=(2))
    vel_r2 = np.random.normal(0, 5, 2)
    #pos_r1 = np.array([100,100])
    #vel_r1 = np.array([0,0])
    #pos_r2 = np.array([105,105])
    #vel_r2 = np.array([-5,-5])
    #crate robots 
    r1 = Robot(pos=pos_r1, vel=vel_r1, noise_std=0.2)
    r2 = Robot(pos=pos_r2, vel=vel_r2, noise_std=0.2)
    #create lists to save time, metrcis, and robot positions
    tt = 0
    tm = [tt]
    mrels = [np.inf]
    dseps = [np.inf]
    ttcs = [np.inf]
    print(r1.pos)
    poss_r1 = [pos_r1.tolist()]
    poss_r2 = [pos_r2.tolist()]
    #run loop to generate data, save it to lists
    for i in range(3000):
        dt = 0.01 #np.abs(np.random.normal(0, 0.1, 1))
        tm.append(tm[-1] + dt)
        pos_r1 = r1.read(dt)
        vel_r1 = r1.vel
        pos_r2 = r2.read(dt)
        vel_r2 = r2.vel
        print(f"r1 {pos_r1} r2 {pos_r2}")
        poss_r1.append(pos_r1)
        poss_r2.append(pos_r2)
        mrel_i, dsep_i = mrel(*pos_r1, *pos_r2, *vel_r1, *vel_r2)
        ttc_i = ttc(*pos_r1, *pos_r2, *vel_r1, *vel_r2, tmax=120, min_dist=10)
        mrels.append(mrel_i)
        dseps.append(dsep_i)
        ttcs.append(ttc_i)
        print('***')
    #convert list to arrays and plot separately: positions, mrel, ttc
    fig, ax = plt.subplots(1,4, figsize=(12,6))
    #print(poss_r1)
    poss_r1 = np.vstack(poss_r1)
    poss_r2 = np.vstack(poss_r2)
    #print(f"poss r1 {poss_r1.shape}")
    
    ax[0].plot(poss_r1[:,0], poss_r1[:,1], color='blue',label='r1')
    ax[0].plot(poss_r2[:,0], poss_r2[:,1], color='red',label='r2')
    ax[0].scatter(poss_r1[0,0], poss_r1[0,1], color='blue', marker='*')
    ax[0].scatter(poss_r2[0,0], poss_r2[0,1], color='red',marker='*')
    ax[0].legend()
    tm = np.vstack(tm)
    mrels = np.hstack(mrels)
    dseps = np.hstack(dseps)
    ttcs = np.hstack(ttcs)
    ax[1].plot(tm, dseps, label='dsep')
    ax[1].legend()
    ax[2].plot(tm, mrels, label='mrel')
    ax[2].legend()
    ax[3].plot(tm, ttcs, label='ttc')
    ax[3].legend()
    plt.show()

if __name__=='__main__':
    main()