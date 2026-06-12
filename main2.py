import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from environment import Environment
from trigger import Trigger
from optimizer import PSOOptimizer
from deploy import RelayDeployer
from agent import LightweightPPOAgent

def generate_z_shape_traj(x_min,x_max,y_min,y_max,z,passes,points=100):
    xs = np.linspace(x_min,x_max,passes)
    traj=[]
    for i,x in enumerate(xs):
        ys = np.linspace(y_min,y_max,points) if i%2==0 else np.linspace(y_max,y_min,points)
        for y in ys:
            traj.append((x,y,z))
    return traj

class EnhancedSimulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self._build_env()
        self._build_components()
        self._init_logging()

    def _build_env(self):
        envc = self.cfg['environment']
        grid = int(envc['area_size']/envc['cell_size'])
        np.random.seed(envc['seed'])
        dsm = np.zeros((grid,grid))
        ob = envc['obstacles']
        for _ in range(ob['num_blocks']):
            x,y = np.random.randint(0,grid-50,2)
            w,h = np.random.randint(30,100,2)
            dsm[x:x+w,y:y+h] = np.random.uniform(ob['min_height'], ob['max_height'])
        self.env = Environment(dsm_map=dsm, cell_size=envc['cell_size'])

        trajc = self.cfg['trajectories']
        area = envc['area_size']
        self.uav1 = generate_z_shape_traj(0,area/2,0,area, trajc['altitude'], trajc['num_passes'])
        self.uav2 = generate_z_shape_traj(area/2,area,0,area, trajc['altitude'], trajc['num_passes'])
        self.steps = min(len(self.uav1), len(self.uav2))
        self.base = tuple(self.cfg['base_station'])

    def _build_components(self):
        self.trigger = Trigger(**self.cfg['trigger'])
        self.optimizer = PSOOptimizer(**self.cfg['optimizer'])
        self.deployer = RelayDeployer(**self.cfg['deployer'])
        self.agent = LightweightPPOAgent(**self.cfg['agent'])
        self.agent.load_model(self.cfg.get('model_path','ppo_model.pth'))

        # 记录
        self.snr_hist = deque(maxlen=10)
        self.comm_imps = deque(maxlen=10)
        self.success_rate = 0.5

    def _init_logging(self):
        self.log_f = open('enhanced_sim_log.csv','w',newline='')
        self.writer = csv.writer(self.log_f)
        self.writer.writerow([
            't','avg_snr_db','total_cap','triggered',
            'sr_mult','pop_mult','update_freq',
            'comm_imp','convergence_speed','reward'
        ])

        self.metrics = {
            'trigger_count':0,
            'comm_improvement':[],
            'convergence_speed':[],
            'rewards':[]
        }

    def run(self):
        relay_traj=[]
        for t in range(self.steps):
            p1, p2 = self.uav1[t], self.uav2[t]
            relay = self.deployer.get_position()

            snrs = [10*np.log10(self.env.get_snr(u,relay)) for u in (p1,p2)]
            avg_snr = float(np.mean(snrs))
            caps = [min(self.env.get_capacity(u,relay), self.env.get_capacity(relay,self.base))
                    for u in (p1,p2)]
            total_cap = float(sum(caps))

            # 更新历史
            self.snr_hist.append(avg_snr)

            # 触发判断
            trig = (t==0) or self.trigger.should_trigger(avg_snr, total_cap, t)
            if trig:
                self.metrics['trigger_count'] += 1

                # 优化前容量
                prev_cap = total_cap

                # 状态构建
                env_info = {
                    'snr_history': list(self.snr_hist),
                    'convergence_speed': np.mean(list(self.comm_imps)) if self.comm_imps else 0.0,
                    'success_rate': self.success_rate,
                    'comm_improvement': np.mean(list(self.comm_imps)) if self.comm_imps else 0.0
                }
                state = self.agent.extract_state(env_info)

                # PPO 得到动作
                action, logp, val = self.agent.get_action(state)
                params = self.agent.map_action_to_params(action)

                # 应用到 PSO
                self.optimizer.num_particles = int(self.cfg['optimizer']['num_particles'] * params['population_multiplier'])
                self.optimizer.num_particles = max(10, min(50, self.optimizer.num_particles))
                # 这里直接用 params['search_radius_multiplier'] 控制 warm-start 搜索半径等，在 PSO 内部自行处理

                # 执行 PSO
                start = time.time()
                new_target = self.optimizer.optimize(self.env, [p1,p2], self.base, relay)
                opt_time = time.time() - start

                self.deployer.set_target(new_target)

                # 优化后容量
                caps2 = [min(self.env.get_capacity(u,new_target), self.env.get_capacity(new_target,self.base))
                         for u in (p1,p2)]
                new_cap = float(sum(caps2))
                comm_imp = new_cap - prev_cap
                self.comm_imps.append(comm_imp)

                # Reward
                convergence_speed = comm_imp / max(opt_time,1)
                reward = self.agent.compute_reward(comm_imp, convergence_speed)
                self.metrics['comm_improvement'].append(comm_imp)
                self.metrics['convergence_speed'].append(convergence_speed)
                self.metrics['rewards'].append(reward)

                self.agent.store_transition(state, action, logp, val, reward, False)
                # PPO 更新
                if self.metrics['trigger_count'] % params['update_frequency'] == 0:
                    self.agent.update()
            else:
                # 未触发时填 0
                params = {'search_radius_multiplier':1.0, 'population_multiplier':1.0, 'update_frequency':1}
                comm_imp = 0.0
                convergence_speed = 0.0
                reward = 0.0

            # 日志
            self.writer.writerow([
                t, avg_snr, total_cap, int(trig),
                params['search_radius_multiplier'],
                params['population_multiplier'],
                params['update_frequency'],
                comm_imp, convergence_speed, reward
            ])

            # 更新中继轨迹
            new_pos = self.deployer.update(dt=1.0)
            relay_traj.append(new_pos)

        self.log_f.close()
        self.agent.save_model(self.cfg.get('model_path','ppo_model.pth'))
        print("✅ 仿真完成，日志保存在 enhanced_sim_log.csv")
        return relay_traj, self.metrics['rewards']

    def save_results(self, traj, rewards):
        out = {
            'trigger_count': self.metrics['trigger_count'],
            'avg_comm_improvement': float(np.mean(self.metrics['comm_improvement'])),
            'avg_convergence_speed': float(np.mean(self.metrics['convergence_speed'])),
            'avg_reward': float(np.mean(rewards))
        }
        with open('simulation_results.json','w') as f:
            json.dump(out, f, indent=2)
        print("📈 统计结果已保存 simulation_results.json")

    def visualize(self, traj, rewards):
        traj = np.array(traj)
        plt.figure(figsize=(6,6))
        u1=np.array(self.uav1); u2=np.array(self.uav2)
        plt.plot(u1[:,0],u1[:,1],label='UAV1')
        plt.plot(u2[:,0],u2[:,1],label='UAV2')
        plt.plot(traj[:,0],traj[:,1],'--',label='Relay')
        plt.legend(); plt.title('Trajectories'); plt.axis('equal'); plt.grid()
        plt.savefig('trajectories.png',dpi=300)
        plt.show()

        plt.figure(figsize=(6,4))
        plt.plot(self.metrics['comm_improvement'],label='Comm Improvement')
        plt.plot(self.metrics['convergence_speed'],label='Convergence Speed')
        plt.plot(rewards,label='Rewards')
        plt.legend(); plt.title('Performance'); plt.grid()
        plt.savefig('performance.png',dpi=300)
        plt.show()

def create_default_config():
    return {
        'environment': {
            'area_size': 1000, 'cell_size':1.0, 'seed':42,
            'obstacles':{'num_blocks':15,'min_height':50,'max_height':120}
        },
        'trajectories':{'altitude':80,'num_passes':5},
        'base_station':[0,0,10],
        'trigger':{
            'avg_snr_thresh':0.9,'snr_fluct_thresh':3.0,
            'avg_cap_thresh':0.9,'cap_fluct_thresh':2.0,'time_interval':50
        },
        'optimizer':{
            'num_particles':30,'num_iterations':50,
            'w_max':0.8,'w_min':0.3,'c1':2.0,'c2':2.0,
            'z_min':20.0,'z_max':120.0,
            'penalty_coeff':100.0,'move_penalty_coeff':0.05,
            'seed':42
        },
        'deployer':{
            'init_pos':(10,10,50),'max_speed_xy':10.0,'max_speed_z':2.0,
            'bounds':(0,0,10,1000,1000,150)
        },
        'agent':{
            'state_dim':4,'action_dim':3,
            'lr':3e-4,'gamma':0.99,'eps_clip':0.2,'k_epochs':3,
            'entropy_coef':0.01,'value_coef':0.5,'max_grad_norm':0.5,
            'device':'cpu'
        },
        'model_path':'ppo_model.pth'
    }

def main():
    print("RLPSOEC: compact PSO+PPO simulation")
    cfg = create_default_config()
    sim = EnhancedSimulator(cfg)
    traj, rewards = sim.run()
    sim.save_results(traj, rewards)
    sim.visualize(traj, rewards)

if __name__ == "__main__":
    main()
