@echo off
if not exist slurm_files (
    mkdir slurm_files
)
scp -o ProxyCommand="ssh -W %%h:%%p bastion" zhaonanmeng@delftblue:~/q_3body_wave_hpc/build/*.out slurm_files\
