## This script is used to invoke the Agent. It doesn't take very large memory or CPU resources but it needs to hang for a long time.
srun --partition=venkvis-cpu --ntasks=1 --nodes=1 --mem=1G --cpus-per-task=1 --time=48:00:00 --pty bash
