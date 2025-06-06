# DDPG


A somewhat modified DDPG algorithm set up to train a model to finish the 2d bipedal walker environment. 

![ezgif-84f6e4714c2c19](https://github.com/user-attachments/assets/4e39b8c1-ea97-4467-a9ff-2e944ad3a553)


## Changes from original DDPG

### Mean Absolute Loss
Instead of mean squared loss mean absolute loss is used since it seemed to drastically improve performance. This idea is based off of [samkoesnadi's](https://github.com/samkoesnadi/DDPG-tf2) DDPG implementation

### Noise Chance Decay
Noise has a chance to be applied rather than being applied at every action.  
