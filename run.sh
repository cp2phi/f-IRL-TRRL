export PYTHONPATH=${PWD}:$PYTHONPATH
# UNIT TEST

# REACHER GOAL REACHING 
#✅ python firl/irl_density.py configs/density/reacher_trace_gauss.yml
#✅ python firl/irl_density.py configs/density/reacher_trace_mix.yml
#✅ python baselines/main_density.py configs/density/reacher_trace_gauss.yml
#✅ python baselines/main_density.py configs/density/reacher_trace_mix.yml

# POINTMASS DOWNSTREAM
#✅ python firl/irl_density.py configs/density/grid_uniform.yml

# 没有continuous这个文件夹
#❌ python continuous/adv_irl/main.py continuous/configs/RSS/reacher_trace_gauss.yml

# MUJOCO IRL BENCHMARK
#✅ python common/train_expert.py configs/samples/experts/hopper.yml
#✅ python firl/irl_samples.py configs/samples/agents/hopper.yml
#✅ python baselines/main_samples.py configs/samples/agents/hopper.yml

# 没有 expert_data/optimal_policy这个文件
#❌ python common/train_optimal.py configs/samples/experts/halfcheetah.yml

# 新版本没有log_prob_unclipped这个函数
#❌ python baselines/bc.py configs/samples/agents/hopper.yml

#KL
#python firl/irl_samples.py configs/samples/agents/ant.yml
#python firl/irl_samples.py configs/samples/agents/halfcheetah.yml
#python firl/irl_samples.py configs/samples/agents/hopper.yml
#python firl/irl_samples.py configs/samples/agents/walker2d.yml

# GAIL
#python imitation/train_imitation.py configs/imitation/ant_GAIL.yml
#python imitation/train_imitation.py configs/imitation/ant_AIRL.yml
#python imitation/train_imitation.py configs/imitation/ant_BC.yml
#python imitation/train_imitation.py configs/imitation/ant_Dagger.yml


python firl/irl_samples.py configs/samples/agents/pendulum_js.yml

python imitation/train_imitation.py configs/imitation/walker2d_GAIL.yml