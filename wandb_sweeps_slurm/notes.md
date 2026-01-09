
### Creating and running a wandb sweep


I ran wandb sweep in slurm by:

1- Creating a sweep and specifying a yaml file:

``` bash
wandb sweep sweep_slurm.yaml
```

This step will provide a sweep id, e.g.: 3m7tpdtv
The yaml file specifies that the agent should run the script called `run_sweep_job.sh`

2- Calling the agent with a number of runs:

``` bash
bash start_sweep_agent.sh
```

The agent will call `run_sweep_job.sh` and submit each run as a job