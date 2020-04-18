# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 03:25:03 2020

@author: Santiago R
"""


import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor



class FeatureTransformer:
    def __init__(self,env,n_components=500):
        obs_ex=np.array([env.observation_space.sample() for x in range(10000)])
        scaler=StandardScaler()
        scaler.fit(obs_ex)
    
        featurizer=FeatureUnion([
            ("rbf1",RBFSampler(gamma=5.0,n_components=n_components)),
            ("rbf2",RBFSampler(gamma=2.0,n_components=n_components)),
            ("rbf3",RBFSampler(gamma=1.0,n_components=n_components)),
            ("rbf4",RBFSampler(gamma=0.5,n_components=n_components))
            ])
        #Converts a state to feature representation
        ex_feat=featurizer.fit_transform(scaler.transform(obs_ex))
        
        self.dimensions=ex_feat.shape[1]
        self.scaler=scaler
        self.featurizer=featurizer
        
    def transform(self,obs):
        scaled=self.scaler.transform(obs)
        return self.featurizer.transform(scaled)
    
class Model:
    def __init__(self,env,feature_transformer,learning_rate):
        self.env=env
        self.models=[]
        self.feature_transformer=feature_transformer
        for i in range(env.action_space.n):
            model=SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()]),[0])
            self.models.append(model)
            
    def predict(self,s):
        X=self.feature_transformer.transform([s])
        result=np.stack([m.predict(X) for m in self.models]).T
        assert(len(X.shape)==2)
        return result
    
    def update(self,s,a,G):
        X=self.feature_transformer.transform([s])
        assert(len(X.shape)==2)
        self.models[a].partial_fit(X, [G])
        
    def sample_actn(self,s,eps):
        if np.random.random()<eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))
        
def play_ep(model,env,eps,gamma):
    obs=env.reset()
    done=False
    totalreward=0
    iters=0
    while not done and iters <10000:
        #env.render()
        action=model.sample_actn(obs,eps)
        prev_obs=obs
        obs, rew, done, info = env.step(action)
        next=model.predict(obs)
        G=rew+gamma*np.max(next[0])
        model.update(prev_obs,action,G)
        totalreward+=rew
        iters +=1
    
    return totalreward

def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    # both X and Y will be of shape (num_tiles, num_tiles)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
    # Z will also be of shape (num_tiles, num_tiles)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z,
      rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("Cost-To-Go Function")
    fig.colorbar(surf)
    plt.show()
  
def plot_running_avg(totalrewards):
    N=len(totalrewards)
    ra=np.empty(N)
    for t in range(N):
        ra[t]=totalrewards[max(0,t-100):t+1].mean()
    plt.plot(ra)
    plt.title("Running Average")
    plt.show()
    
def main(show_plots=True):
    env=gym.make("MountainCar-v0")
    ft=FeatureTransformer(env)
    model=Model(env,ft,"constant")
    gamma=0.99
    
    if "monitor" in sys.argv:
        filename=os.path.basename(__file__).split(".")[0]
        monitor_dir="c:\\Users\\Santiago R\\mountaincar"
        env=wrappers.Monitor(env,monitor_dir,force=True)
        
    N=300
    totalrewards=np.empty(N)
    for n in range(N):
        eps=0.1*(0.97**n)
        if n==199:
            print("eps:", eps)
        totalreward=play_ep(model, env, eps, gamma)
        totalrewards[n]=totalreward
        if (n+1)%100==0:
            print("episodes:",n," totalreward:", totalreward)
    print("Avg. reward for last 100:", totalrewards[-100:].mean())
    print("Total steps:",-totalrewards.sum())
    
    if show_plots:
        plt.plot(totalrewards)
        plt.title("Rewards")
        plt.show()
        plot_running_avg(totalrewards)
        plot_cost_to_go(env, model)
        
if __name__=="__main__":
    main()
    