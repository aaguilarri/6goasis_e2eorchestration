import numpy as np
import pandas as pd
from numpy.random import uniform
from numpy.random import normal
import matplotlib.pyplot as plt
import scipy
from matplotlib.animation import FuncAnimation
import time
import math
import randomname
import copy
from scipy.stats import multivariate_normal
import pandas as pd
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

class Robot:
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.ID = 'Rx_'+randomname.get_name()
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]
        acc = 0.01
    def read(self, dt):
        pos = [0,0]
        self.pos[0] += self.vel[0] *dt 
        self.pos[1] += self.vel[1] *dt 
        rng = np.random.default_rng()
        facts = 0.5 * rng.normal(0, self.noise_std,2) * dt**2
        z0 = self.pos[0] + facts[0]
        z1 = self.pos[1] + facts[1]
        return [z0.item(), z1.item()]
    
def fx(x, dt):
    F = np.array([[1, dt, 0, 0],
                  [0,  1, 0, 0],
                  [0,  0, 1, dt],
                  [0,  0, 0, 1]], dtype=float)
    return F @ x

def hx(x):
    return np.array([ x[0], x[2] ])  # measure x-position and y-position


class Ufilter():
    def __init__(self, z, tck, sp=1000, sq=1, sr=1, typo=None):
        key = None
        vals_dict = {'':[15,15], 'person':[2,1], 'car':[0.1,10], 'car_hero_camera':[10,15], 'car_agent_camera':[2,5], 'bus':[10,15],  'truck':[35,35],'traffic light':[4,0.1], 'fire hydrant':[4,0.1], 'parking meter':[4,0.1], 'stop sign':[4,0.1]}
        if typo is None:
            key = ''
        else:
            key = typo
        self.type = key
        vals = vals_dict[key]
        self.id = 'uf_'+randomname.get_name() #str(int(time.time()*10**7))
        self.last_tck = tck
        self.color = np.random.rand(3,)
        ndim = 4
        self.ndim= ndim
        self.alpha = 0.1
        self.beta = 2.
        self.kappa = -1
        self.dt = 0.001
        self.denshi = np.zeros([ndim, ndim])
        points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)
        self.ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=self.dt ,hx=hx,fx=fx, points=points)
        #self.x = np.array([z[0], np.random.normal(0, vals[0],1).item(), z[1], np.random.normal(0, vals[0],1).item()])
        self.x = np.array([z[0], 0, z[1], 0])
        self.sq = vals[0] #* self.mdeg
        self.sr = vals[1] #* self.mdeg
        self.x = self.x.reshape([ndim, 1])
        self.P = self.getP(sp)
        self.Q = np.zeros([ndim,ndim])
        self.R = self.getR(sigma=self.sr)
        self.ukf.x = self.x.flatten()
        self.ukf.P = self.P
    
    def getH(self):
        H = None
        H = np.array([[1,0,0,0],[0,0,1,0]])
        return H
     
    def getP(self, sigma=1):
        print(f" sigma {sigma}, dim {self.ndim}")
        return sigma * np.eye(self.ndim)
    
    def getQ(self, dt, sigma=1.):
        Q = None
        f0 = np.array([[0.25 * dt**4 , 0.5 * dt**3],[0.5 * dt**3, dt**2]])
        z0 = np.zeros([2,2])
        Q = np.vstack([np.hstack([f0, z0]), np.hstack([z0, f0])])
        return sigma * Q
    
    def getR(self, sigma=1): 
        R = sigma * np.eye(2)
        return R

    def predict(self, dt):
        self.ukf.Q = self.getQ(dt, self.sq)
        self.ukf.predict(dt=dt)
        self.x = self.ukf.x
        self.P = self.ukf.P
        return self
    
    def update(self, z):
        self.ukf.update(z)
        self.x = self.ukf.x
        self.P = self.ukf.P
        return self
    
    def update_ts(self, ts): #need to update ts if working with timestamps
        self.last_tck = ts

    def estimate_pos(self,q):
        return self.x.flatten()
    
    def resample(self):
        return

    def likelihood(self, z, z0,dt):
        return np.exp(self.ukf.log_likelihood)
    
class Kfilter():
    def __init__(self, z, tck, sp=1000, sq=1, sr=1, typo=None):
        ndim = 4
        self.ndim = ndim
        key = None
        vals_dict = {'':[15,15], 'person':[2,1], 'car':[0.1,10], 'car_hero_camera':[10,15], 'car_agent_camera':[2,5], 'bus':[10,15],  'truck':[35,35],'traffic light':[4,0.1], 'fire hydrant':[4,0.1], 'parking meter':[4,0.1], 'stop sign':[4,0.1]}
        if typo is None:
            key = ''
        else:
            key = typo
        vals = vals_dict[key]
        self.mdeg = 0.00000898334580010698 #1 / 111320 #degress / m
        self.sq = vals[0] #* self.mdeg
        self.sr = vals[1] #* self.mdeg
        self.type = key
        self.id = 'kf_'+randomname.get_name() #str(int(time.time()*10**7))
        self.last_tck = tck
        self.color = np.random.rand(3,)
        self.kmtm = 1/3.6 #Km to m
        #self.k = max_dist*self.mdeg #0.0713349 #exp factor k for udpate()
        self.rd = 6371000 #Earth's radius
        #self.x = np.array([np.random.normal(z[0], 1,1).item(), np.random.normal(0, vals[0],1).item() , np.random.normal(z[1], 1,1).item(), np.random.normal(0, vals[0],1).item() ])
        self.x = np.array([z[0], np.random.normal(0, vals[0],1).item(), z[1], np.random.normal(0, vals[0],1).item()])
        #self.x = np.array([z[0],0,z[1], 0])
        self.x = self.x.reshape([ndim, 1])
        self.P = self.getP(sp)
        self.F = np.zeros([ndim,ndim])
        self.Q = np.zeros([ndim,ndim])
        self.H = self.getH()
        self.R = self.getR(sigma=self.sr)
        self.denshi = np.zeros([ndim, ndim])
    
    def getF(self, dt):
        F = None
        f0 = np.array([[1,dt],[0,1]])
        z0 = np.zeros([2,2])
        F = np.vstack([np.hstack([f0, z0]), np.hstack([z0, f0])])
        return F
    
    def getH(self):
        H = None
        H = np.array([[1,0,0,0],[0,0,1,0]])
        return H
    
    def getP(self, sigma=1):
        print(f" sigma {sigma}, dim {self.ndim}")
        return sigma * np.eye(self.ndim)
    
    def getQ(self, dt, sigma=1.):
        Q = None
        f0 = np.array([[0.25 * dt**4 , 0.5 * dt**3],[0.5 * dt**3, dt**2]])
        z0 = np.zeros([2,2])
        Q = np.vstack([np.hstack([f0, z0]), np.hstack([z0, f0])])
        return sigma * Q
    
    def getR(self, sigma=1): 
        R = sigma * np.eye(2)
        return R
    
    def update_ts(self, ts): #need to update ts if working with timestamps
        self.last_tck = ts

    def predict(self, dt):
        x = np.copy(self.x)
        F = self.getF(dt)
        Q = self.getQ(dt)
        P = np.copy(self.P)
        self.x = F @ x
        self.P = F @ (P @ F.T) + Q   
        self.F = np.copy(F)
        self.Q = np.copy(Q)
        return self

    def update(self, z):
        lkl = self.predict_lkl(z)
        if math.isinf(lkl):
            return
        lz = len(z)
        z = np.array(z)
        z = z.reshape([lz,1])
        H = np.copy(self.H)
        x = np.copy(self.x)
        P = np.copy(self.P)
        R = np.copy(self.R)
        y = z - H @ x
        S = H @ (P @ H.T) + R
        Si = np.linalg.inv(S)
        K = P @ (H.T @ Si)
        self.x = x + K @ y
        I = np.eye(self.ndim)
        self.P = (I - K @ H) @ P
        return self
    
    def estimate_pos(self,q):
        return self.x.flatten()
    
    def resample(self):
        return

    def likelihood(self, z, z0,dt):
        lz = len(z)
        z = np.array(z)
        z = z.reshape([lz,1])
        H = np.copy(self.H)
        x = np.copy(self.x)
        P = np.copy(self.P)
        R = np.copy(self.R)
        S = H @ (P @ H.T) + R
        xs = H @ x
        xs = xs.flatten()
        n_dist = multivariate_normal(mean=xs, cov=S)
        lkl = n_dist.pdf(z.flatten())
        return lkl

    def predict_lkl(self, z, my_R=None):
        lz = len(z)
        z = np.array(z)
        z = z.reshape([lz,1])
        H = np.copy(self.H)
        x = np.copy(self.x)
        P = np.copy(self.P)
        R = np.copy(self.R)
        S = H @ (P @ H.T) + R
        xs = H @ x
        xs = xs.flatten()
        n_dist = multivariate_normal(mean=xs, cov=S)
        lkl = n_dist.pdf(z.flatten())
        return lkl

class Pfilter():

    def __init__(self, n, lat, lon, tck, color, max_dist=35, max_vel=35, typo=None):
        vals_dict = {'':[15,15], 'person':[2,1], 'car':[2,1], 'car_hero_camera':[10,15], 'car_agent_camera':[10,15], 'bus':[10,15],  'truck':[35,35],'traffic light':[4,0.1], 'fire hydrant':[4,0.1], 'parking meter':[4,0.1], 'stop sign':[4,0.1]}
        #'car':[10,15]
        key = None
        if typo is None:
            key = ''
        else:
            key = typo
        vals = vals_dict[key]
        max_dist = vals[0]
        max_vel = vals[1]
        self.type = key
        self.id = 'pf_'+randomname.get_name() #str(int(time.time()*10**7))
        self.last_tck = tck
        self.color = color
        self.n = n #number of particles
        self.mdeg = 0.00000898334580010698 #1 / 111320 #degress / m
        self.kmtm = 1/3.6 #Km to m
        self.R = 1 #uncertainty in predict
        self.k = max_dist*self.mdeg #0.0713349 #exp factor k for udpate()
        self.rd = 6371000 #Earth's radius
        self.max_dist = max_dist #maximum distance in m to create particles
        self.max_vel = max_vel #maximum partilce speed km/h
        self.denshi = self.create_particles(self.n, lat, lon)
        self.estimate = [lat, self.denshi[0,1], lon, self.denshi[0,3]]
        self.trajectory = [self.estimate] #list of estimations
    
    def create_particles(self, n, lat0, lon0):
        #n, number of particles
        #m lat0,lon0: point of reference
        #print("Player 1, wecolme to create particles")
        #print(f"reference is {lat0}, {lon0}")
        ddg = self.mdeg # kilometers / degree
        vms = self.kmtm * self.max_vel
        delta = self.max_dist
        ss = ddg * self.max_dist
        dv = vms * ddg # maximum velocity
        #print("---- create_particles()-----")
        #print(f"m/degree = {self.mdeg} ")
        #print(f"max vel = {self.max_vel}km/h /* {self.kmtm} = {self.max_vel*self.kmtm}")
        #print(f"result: {self.mdeg*self.max_vel*self.kmtm}")
        #print(f" dv is {dv}")
        latN = lat0 + delta * ddg
        latS = lat0 - delta * ddg
        lonE = lon0 + delta * ddg
        lonW = lon0 - delta * ddg
        denshi = np.empty((n,5))
        denshi[:,0] = normal(lat0, ss, n)
        denshi[:,1] = normal(0., dv, n)
        denshi[:,2] = normal(lon0, ss, n)
        denshi[:,3] = normal(0., dv, n)
        #denshi[:,0] = uniform(latS, latN, size=n)
        #denshi[:,1] = uniform(-dv, dv, size=n)
        #denshi[:,2] = uniform(lonW, lonE, size=n)
        #denshi[:,3] = uniform(-dv, dv, size=n)
        denshi[:,4] = (1/n) * np.ones(n)
        #print(pd.DataFrame(denshi).to_string)
        return denshi

    def predict(self, denshi, dt):
        ddg = self.mdeg # meters / degree
        max_dist = self.max_dist
        max_vel = self.max_vel
        kmtm = self.kmtm
        nr,nc = denshi.shape
        mt = np.array([[1., dt, 0., 0., 0.], [0, 1, 0., 0., 0.], [0., 0., 1., dt, 0.], [0., 0., 0, 1, 0.], [0., 0., 0., 0., 1.]]).T
        #print(pd.DataFrame(mt).to_string())
        denshi[:,1] += self.max_vel * (ddg)*np.random.uniform(-1,1, size=nr)
        denshi[:,3] += self.max_vel * (ddg)*np.random.uniform(-1,1, size=nr)
        denshi = np.matmul(denshi, mt)
        
        return denshi

    def update(self, denshi, pt, dt):
        #print(f"Update()")
        k = self.k #*self.R
        nr, nc = denshi.shape
        #dist = self.haversine(denshi[:,0], denshi[:,2], pt[0]*np.ones(nr), pt[1]*np.ones(nr))
        locs = np.vstack([denshi[:,0],denshi[:,2]]).T
        dlats = np.square(denshi[:,0] - pt[0]*np.ones(nr))
        dlons = np.square(denshi[:,2] - pt[1]*np.ones(nr))
        dist =  np.sqrt(dlats+dlons)
        k += np.std(dist)
        norms = scipy.stats.norm(dist, k).pdf(0.)
        if np.sum(norms) <= 0:
            return denshi, False
        #dist = self.haversine(denshi[:,0], denshi[:,2], pt[0]*np.ones(nr), pt[1]*np.ones(nr))
        #norms = np.exp(-k*dist)
        #print(f"norms {norms}")
        ws = denshi[:,-1]
        ws *= norms
        ws /= np.sum(ws)
        denshi[:,-1] = ws
        Nef = 1/self.n
        denshir = denshi[denshi[:,-1] > Nef,:]
        denshir[:,-1] /= np.sum(denshir[:,-1])
        print(f"denshi ris size {denshir.shape}")
        return denshir, True

    def resample2(self, denshi, z):
        #print(f"Resample2()")
        pos =  self.estimate_pos(denshi).flatten()
        denshi2 = self.create_particles(self.n, pos[0], pos[1])
        udenshi, ok = self.update(denshi2, z, 0)
        #print(f"udenshi {udenshi.shape}")
        ws0 = denshi[:,-1]
        ws1 = udenshi[:,-1]
        dif = ws0 > ws1
        mask = dif.astype(int)
        mask = mask[:,np.newaxis]
        the_den = mask * denshi + (1 - mask) * udenshi
        the_den[:,-1] /= np.sum(the_den[:,-1])
        return the_den

    def resample(self, denshi, estimate):
        nmax = self.n
        cumulative_sum = np.cumsum(denshi[:,-1])
        cumulative_sum[-1] = 1.0  # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.random(nmax))
        denshi2 = denshi[indexes]
        denshi2[:,-1] /= np.sum(denshi2[:,-1])
        #print(f"indexes->{indexes}")
        return denshi2

        pj = estimate
        denshi1 = np.random.permutation(denshi)
        mask = denshi[:,-1] > denshi1[:,-1]
        denshi_up = denshi[mask,:]
        denshi = np.vstack([denshi_up, denshi_up])
        denshi = np.random.permutation(denshi)
        limi = nmax // 10
        denshi = denshi[:limi, :]
        nd = denshi.shape[0]
        if nd > nmax:
            denshi = denshi[0:nmax,:]
        if nd < nmax:
            diffd = nmax - nd
            denshij = self.create_particles(diffd, pj[0], pj[1])
            denshi = np.vstack([denshi, denshij])
        denshi[:,-1] /= np.sum(denshi[:,-1])
        return denshi
    
    def estimate_pos(self, denshi):
        #print(denshi)
        ws = denshi[:,-1]
        ws = ws[:,np.newaxis]
        xi = np.matmul(denshi.T, ws)
        #print(f"xi = {xi}")
        return xi   

    def filtering(self, sample, dt):
        denshi = self.predict(self.denshi, dt)
        denshi, ok = self.update(denshi, sample, dt)
        denshi = self.resample(denshi, sample)
        self.estimate = self.estimate_pos(denshi)
        #self.plot_state(denshi, sample, self.estimate, dt)
        return denshi
    
    def likelihood(self, pt, pt0,  dt):
        #print(f"pt={pt}, pt0={pt0}, dt={dt}")
        k = self.k #*self.R
        denshi = self.predict(self.denshi,dt)
        nr, nc = denshi.shape
        dists = np.array([denshi[:,0] - pt[0], denshi[:,2]-pt[1]]).T
        #print(dists.shape)
        dists = np.sqrt(dists[:,0]*dists[:,0] + dists[:,1]*dists[:,1])
        mdist = np.mean(dists)
        sdist = np.std(dists)
        #dptdt = [(pt[0]-pt0[0])/dt, (pt[1]-pt0[1])/dt]
        #print("Welcome to Likelihood!")
        #print(f"Pos is {pt}")
        #print(f"Pos0 is {pt0}")
        #print(f"dt is {dt}")
        #print(f"Estimated speed is {dptdt}")
        #mts = haversine(pt[0],pt[1],pt0[0],pt0[1])
        #spd = mts / dt
        lkl = scipy.stats.norm(mdist, sdist).pdf(0.)
        #print("************************")
        #print("Welcome to Likelihood!")
        #print(f"Pos is {pt}")
        #print(f"Pos0 is {pt0}")
        #print(f"dt is {dt}")
        #print(f"dist is {dist}")
        #print(f"speed is {speed}")
        #print(f"mdist is {mdist}")
        #print(f"sdist is {sdist}")
        #print(f"stds is {stds}")
        #print(f"likelihood is {lkl}")
        if np.isnan(lkl):
            lkl = 0
        return lkl
        #dist = self.haversine(denshi[:,0], denshi[:,2], pt[0]*np.ones(nr), pt[1]*np.ones(nr))
        #locs = np.vstack([denshi[:,0],denshi[:,2]]).T
        dlats = np.square(denshi[:,0] - pt[0]*np.ones(nr))
        dlatsdt = np.square(denshi[:,1] - dptdt[0]*np.ones(nr))
        dlons = np.square(denshi[:,2] - pt[1]*np.ones(nr))
        dlonsdt = np.square(denshi[:,3] - dptdt[1]*np.ones(nr))
        #lats = np.vstack([denshi[:,0]*10**3,denshi[:,2]*10**3]).T
        #meanos = np.mean(dats,axis=0)
        #covs = np.cov(dats, rowvar=False)
        #print(f"meanos = {meanos}")
        #print(f"covs = {covs}")
        #sample= np.array([pt[0],pt[1]])
        ##mvn = scipy.stats.multivariate_normal(mean=meanos,cov=covs)
        #lkl = mvn.pdf(sample)
        #print(f"lkl = {lkl}")
        #print("**** bye! *****")
        #return lkl
        dist =  np.sqrt(dlats+dlons+dlatsdt+dlonsdt)
        mdist= np.mean(dist)
        k = np.std(dist)
        #print(f"mdist = {mdist}, k={k}")
        norms = scipy.stats.norm(mdist, k).pdf(0.)
        #print(f"lkl = {norms}")
        return norms
        #dist = self.haversine(denshi[:,0], denshi[:,2], pt[0]*np.ones(nr), pt[1]*np.ones(nr))
        #norms = np.exp(-k*dist)
        ws = denshi[:,-1]
        ws *= norms
        ws /= np.sum(ws)
        denshi = self.denshi
        nr, nc = denshi.shape
        hvrs = []
        for i in range(nr):
            hvr = self.haversine(pt[0], pt[1], denshi[i,0], denshi[i,2])
            hvrs.append(hvr)
        meano = np.mean(hvrs)
        stdo = np.std(hvrs)
        norms = scipy.stats.norm.pdf(0., loc=meano, scale=stdo) 
        #print(f"meano = {meano}")
        #print(f"sdev = {stdo}")
        #print(f"norms = {norms}")
        return norms
        locs = np.hstack([denshi[:,0],denshi[:,2]])
        dlats = np.square(denshi[:,0] - pt[0]*np.ones(nr))
        dlons = np.square(denshi[:,2] - pt[1]*np.ones(nr))
        dists =  np.sqrt(dlats+dlons)
        sdev = np.std(dists)
        meano = np.mean(dists)
        norms = scipy.stats.norm.pdf(0., loc=meano, scale=sdev)
        #print(f"dlats={dlats}, dlons={dlons}, dists={dists}")
        #print(f"meano = {meano}")
        #print(f"sdev = {sdev}")
        #print(f"norms = {norms}")
        return norms

    def haversine(self, lat1, lon1, lat2, lon2):
        # Haversine formula (distance between lats, lons)
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        radius = self.rd  # approximately 6,371 km
        distance = radius * c
        return distance
    

    def plot_state(self, denshi, sample, estado, dt, past=None):
        fig, ax = plt.subplots()
        ax.set_facecolor('black')
        cent = past[0][0]
        #print(sample)
        latN = cent[0] + self.mdeg * 300
        latS = cent[0] - self.mdeg * 300
        lonE = cent[1] + self.mdeg * 300
        lonW = cent[1] - self.mdeg * 300
        plt.xlim(latS, latN)
        plt.ylim(lonW, lonE)
        # Plot all three lines
        ax.plot(denshi[:, 0], denshi[:, 2], marker='+', linestyle='', label='Line 1')
        ax.plot(sample[0], sample[1], marker='o', color='red', linestyle='', label='Point 1')
        ax.plot(estado[0], estado[1], marker='*', color='yellow', linestyle='', label='Point 2')
        
        if past is not None:
            pasto = np.array(past)
            pasto = np.vstack(pasto)
            ax.plot(pasto[:, 0], pasto[:, 1], marker='.', color='green', linestyle='', label='Line 2')
        plt.title(f"Delta time is {dt}")
        #plt.pause(1)
        #plt.close()
        
        plt.show()
        #plt.close()
        #plt.hist(denshi[:,-1])
        #plt.show()
        return ax

def main():
    robo = Robot(pos=(100,100), vel=(1,1), noise_std=10)
    xs = []
    ys = []
    zs = []
    dt = 0.1
    tmax = 5
    tss = np.arange(0,tmax,dt)
    ukf = None
    for ts in tss:
        #print(f"ts {ts}, dt {dt}")
        if ts < dt:
            ukf = Ufilter([100,100],tss[0])
            #print(f"filter {ukf.id} created")
            #print('*****')
            continue
        dt = ts - ukf.last_tck
        print('.', end='', flush=True)
        z = robo.read(dt)
        #print(f"dt {dt}")
        #print(f" z {z}")
        #print(f" y {np.array(robo.pos).flatten()}")
        #print(f"before, x {ukf.x.flatten()}")
        ys.append(np.array(robo.pos).flatten())
        zs.append(np.array(z).flatten())
        ukf.predict(dt)
        #print(f"after predict, x {ukf.x.flatten()}")
        ukf.update(z)
        #print(f"after update, x {ukf.x.flatten()}")
        xs.append(ukf.x)
        ukf.last_tck = ts
        #print("*****")
    print("")
    print("Done.")
    xs = np.vstack(xs)
    ys = np.vstack(ys)
    zs = np.vstack(zs)
    fig, ax = plt.subplots(1,1)
    ax.plot(xs[:,0], xs[:,2], color='green')
    ax.plot(ys[:,0], ys[:,1], color='blue')
    ax.plot(zs[:,0], zs[:,1], color='red')
    plt.show()
    exit()
    n = 1000
    latN = None
    latS = None
    LonE = None
    LonW = None
    ddg = 0.00000898334580010698
    delta = 200
    poss = []
    pathos = '../edge_files_full/edge_files/'
    fname = 'car_data.csv'
    df = pd.read_csv(pathos+fname)
    #for i in range(50):
    filt = None
    fig = None
    ax = None
    for ix, row in df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        t1 = row['timestamp']
        if filt is None:
            filt = Pfilter(n,lat,lon,t1,np.random.rand(3,),typo='car')
            latN = lat + delta * ddg
            latS = lat - delta * ddg
            lonE = lon + delta * ddg
            lonW = lon - delta * ddg
            fact = 30*ddg/3.6

            continue
        t0 = filt.last_tck
        dt = t1 - t0
        print('.', end='', flush=True)
        den = copy.copy(filt.denshi)
        den = filt.predict(den, dt)
        den, ok = filt.update(den, [lat, lon], 0)
        w = den[:,-1]
        weff = 1 / np.sum(w**2)
        print(f"weff is {weff}")
        if weff < n/2:
            den = filt.resample(den,[lat,lon])
        pos = filt.estimate_pos(den)
        filt.denshi = copy.copy(den)
        filt.last_tck = t1
        parts = filt.denshi
        poss.append([lat, lon])
        pj = [pos[0], pos[2]]
        pi= np.vstack(poss)
        

        fig, ax = plt.subplots()
        ax.set_facecolor('black')
        plt.xlim(latS, latN)
        plt.ylim(lonW, lonE)
        # Plot all three lines

        ax.plot(parts[:, 0], parts[:, 2], marker='+', linestyle='', label='Line 1')
        ax.plot(pi[:,0], pi[:,1], marker='o', color='red', linestyle='', label='Point 1')
        ax.plot(pj[0], pj[1], marker='*', color='yellow', linestyle='', label='Point 2')
        

        plt.title(f"Delta time is {t1 - t0}")
        plt.legend()  # Add legend to differentiate lines/points
        plt.pause(1)
        plt.close()
    plt.show()
    plt.savefig('car.png')
    exit()

    lat = 41.274927054998706
    lon = 1.9865927764756361
    t0 = time.time()
    
    ddg = filt.mdeg
    delta = 2000
    latN = lat + delta * ddg
    latS = lat - delta * ddg
    lonE = lon + delta * ddg
    lonW = lon - delta * ddg
    fact = 30*filt.mdeg/3.6
    for i in range(50):
        lat += fact
        lon -= fact
        t1 = time.time()
        dt = t1 - t0
        print(f"dt = {dt}")
        den = copy.copy(filt.denshi)
        den = filt.predict(den, dt)
        den, ok = filt.update(den, [lat, lon], 0)
        w = den[:,-1]
        weff = 1 / np.sum(w**2)
        if weff < n/2:
            den = filt.resample(den,[lat,lon])
        pos = filt.estimate_pos(den)
        filt.denshi = copy.copy(den)
        filt.last_tck = t1
        parts = filt.denshi
        poss.append([lat, lon])
        pj = [pos[0], pos[2]]
        pi= np.vstack(poss)
        

        fig, ax = plt.subplots()
        ax.set_facecolor('black')
        plt.xlim(latS, latN)
        plt.ylim(lonW, lonE)
        # Plot all three lines

        ax.plot(parts[:, 0], parts[:, 2], marker='+', linestyle='', label='Line 1')
        ax.plot(pi[:,0], pi[:,1], marker='o', color='red', linestyle='', label='Point 1')
        ax.plot(pj[0], pj[1], marker='*', color='yellow', linestyle='', label='Point 2')
        

        plt.title(f"Delta time is {t1 - t0}")
        plt.legend()  # Add legend to differentiate lines/points
        plt.pause(0.1)
        t0 = t1
        
        plt.close()
    plt.show()



if __name__ == '__main__':
    main()